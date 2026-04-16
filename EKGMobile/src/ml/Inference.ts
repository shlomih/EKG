/**
 * Inference.ts — Full ECG inference pipeline.
 *
 * Takes raw 12-lead signal + patient demographics, runs:
 *   1. Preprocessing (resample, normalize, feature extraction)
 *   2. ONNX model inference
 *   3. Temperature scaling (per-class)
 *   4. Sigmoid + threshold comparison
 *   5. Returns detected conditions sorted by urgency
 *
 * Source: multilabel_v3.py predict_v3() lines 760-831
 */

import { preprocess, N_AUX, SIGNAL_LEN, N_LEADS } from './Preprocessor';
import { toDetectedCondition, type DetectedCondition } from '@/src/types/Analysis';
import { V3_CODES, V3_URGENCY, type ConditionCode } from './ConditionMetadata';
import { logEvent } from '@/src/audit/AuditLogger';

/**
 * Per-class temperature scaling values from models/thresholds_v3.json.
 * These are fitted on the validation set to improve probability calibration.
 * Order matches V3_CODES exactly.
 */
const TEMPERATURES: readonly number[] = [
  1.11069655418396,     // NORM
  2.04852557182312,     // AFIB
  2.0016205310821533,   // PVC
  1.739318609237671,    // LVH
  2.195678949356079,    // IMI
  2.1193768978118896,   // ASMI
  1.6561954021453857,   // CLBBB
  1.5772089958190918,   // CRBBB
  1.8851035833358765,   // LAFB
  1.8277024030685425,   // 1AVB
  1.9064425230026245,   // ISC_
  2.155794143676758,    // NDT
  2.0671944618225098,   // IRBBB
  1.727296233177185,    // STACH
  1.9541736841201782,   // PAC
  1.1966749429702759,   // Brady
  1.5031179189682007,   // SVT
  1.9691084623336792,   // LQTP
  1.8138694763183594,   // TAb
  2.0029733180999756,   // LAD
  2.133449077606201,    // RAD
  1.9339721202850342,   // NSIVC
  1.705942153930664,    // AFL
  2.071763038635254,    // STc
  2.0525665283203125,   // STD
  2.0693600177764893,   // LAE
] as const;

/**
 * Per-class thresholds from models/thresholds_v3.json.
 * Probabilities >= threshold => condition detected.
 * Order matches V3_CODES exactly.
 */
const THRESHOLDS: readonly number[] = [
  0.47245525840855535,  // NORM
  0.7487373519555263,   // AFIB
  0.8795209974884366,   // PVC
  0.7789220503320698,   // LVH
  0.9340977521859598,   // IMI
  0.7709158021544179,   // ASMI
  0.8699197212132893,   // CLBBB
  0.8557985214201106,   // CRBBB
  0.8418726294302578,   // LAFB
  0.8872930679414802,   // 1AVB
  0.893792642750716,    // ISC_
  0.8895855883983328,   // NDT
  0.9255481594416122,   // IRBBB
  0.8986686827305226,   // STACH
  0.7752691083214058,   // PAC
  0.7852624575331191,   // Brady
  0.31753514834966357,  // SVT
  0.9232420345132245,   // LQTP
  0.7229763492563419,   // TAb
  0.8194401646678886,   // LAD
  0.7714187653436497,   // RAD
  0.8989777185669504,   // NSIVC
  0.7543146562863315,   // AFL
  0.835558428807455,    // STc
  0.8216161872802145,   // STD
  0.95,                 // LAE
] as const;

const N_CLASSES = V3_CODES.length; // 26

/**
 * Inference result with full probability information.
 */
export interface InferenceResult {
  /** Detected conditions (above threshold), sorted by urgency desc then probability desc */
  conditions: DetectedCondition[];
  /** Primary condition (highest urgency detected, or highest probability if none) */
  primary: string;
  /** Raw probabilities for all 26 classes (after temperature scaling + sigmoid) */
  scores: Record<string, number>;
  /** Model version string */
  modelVersion: string;
}

/**
 * Sigmoid function with numerical stability.
 * For large negative x, exp(-x) overflows, so we use the identity:
 *   sigmoid(x) = x >= 0 ? 1/(1+exp(-x)) : exp(x)/(1+exp(x))
 */
function sigmoid(x: number): number {
  if (x >= 0) {
    return 1.0 / (1.0 + Math.exp(-x));
  }
  const ex = Math.exp(x);
  return ex / (1.0 + ex);
}

/**
 * Run inference on raw 12-lead ECG signal.
 *
 * @param signal    Array of 12 Float32Arrays (one per lead, raw mV values)
 * @param session   ONNX InferenceSession (from ModelManager)
 * @param sex       Patient sex: 'M' or 'F'
 * @param age       Patient age in years
 * @returns InferenceResult with detected conditions and scores
 */
export async function runInference(
  signal: Float32Array[],
  session: { run: (feeds: Record<string, { data: Float32Array; dims: number[] }>) => Promise<Record<string, { data: Float32Array }>> },
  sex: 'M' | 'F' = 'M',
  age: number = 50,
): Promise<InferenceResult> {
  // Step 1: Preprocess
  const { signal: signalTensor, aux: auxTensor } = preprocess(signal, sex, age);

  // Step 2: Run ONNX inference
  const feeds = {
    signal: { data: signalTensor, dims: [1, N_LEADS, SIGNAL_LEN] },
    aux: { data: auxTensor, dims: [1, N_AUX] },
  };

  const output = await session.run(feeds);
  const logits = output['logits']!.data;

  // Step 3: Apply per-class temperature scaling + sigmoid
  const probs = new Float32Array(N_CLASSES);
  for (let i = 0; i < N_CLASSES; i++) {
    const calibratedLogit = logits[i]! / TEMPERATURES[i]!;
    probs[i] = sigmoid(calibratedLogit);
  }

  // Step 4: Compare to thresholds, build detected conditions
  const scores: Record<string, number> = {};
  const detected: DetectedCondition[] = [];

  for (let i = 0; i < N_CLASSES; i++) {
    const code = V3_CODES[i]!;
    const prob = probs[i]!;
    scores[code] = prob;

    if (prob >= THRESHOLDS[i]!) {
      detected.push(toDetectedCondition(code, prob));
    }
  }

  // Step 5: Sort by urgency (descending) then probability (descending)
  detected.sort((a, b) => {
    if (b.urgency !== a.urgency) return b.urgency - a.urgency;
    return b.probability - a.probability;
  });

  // Primary: highest urgency detected condition, or highest probability overall
  let primary: string;
  if (detected.length > 0) {
    primary = detected[0]!.code;
  } else {
    let maxIdx = 0;
    let maxProb = probs[0]!;
    for (let i = 1; i < N_CLASSES; i++) {
      if (probs[i]! > maxProb) {
        maxProb = probs[i]!;
        maxIdx = i;
      }
    }
    primary = V3_CODES[maxIdx]!;
  }

  await logEvent('INFERENCE_RUN', 'ecg_inference', null, null,
    `conditions:${detected.length},primary:${primary}`);

  return {
    conditions: detected,
    primary,
    scores,
    modelVersion: 'V3.2b',
  };
}
