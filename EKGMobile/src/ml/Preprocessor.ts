/**
 * Preprocessor.ts — Prepare raw 12-lead ECG signal for ONNX inference.
 *
 * Pipeline (matches Python multilabel_v3.py predict_v3):
 *   1. Resample to 5000 samples if needed (linear interpolation)
 *   2. Normalize signal by /5.0 (amplitude-preserving, NOT z-score)
 *   3. Extract voltage features (14) + RR features (4) = 18 aux features
 *   4. Return typed arrays ready for ONNX: signal(1,12,5000) + aux(1,18)
 *
 * Source: cnn_classifier.py GLOBAL_NORM_SCALE=5.0, multilabel_v3.py predict_v3()
 */

import { extractVoltageFeatures } from './VoltageFeatures';
import { extractRRFeatures } from './RRFeatures';

/** Expected signal length after resampling (10s at 500Hz) */
export const SIGNAL_LEN = 5000;
/** Number of leads */
export const N_LEADS = 12;
/** Number of auxiliary features: 14 voltage + 4 RR */
export const N_AUX = 18;
/** Normalization divisor (matches Python GLOBAL_NORM_SCALE) */
const NORM_SCALE = 5.0;

/**
 * Preprocessed tensors ready for ONNX inference.
 */
export interface PreprocessedInput {
  /** Signal tensor, shape [1, 12, 5000], float32 */
  signal: Float32Array;
  /** Auxiliary features tensor, shape [1, 18], float32 */
  aux: Float32Array;
}

/**
 * Resample a single-lead signal from srcLen to dstLen using linear interpolation.
 */
function resampleLead(src: Float32Array, dstLen: number): Float32Array {
  const srcLen = src.length;
  if (srcLen === dstLen) return new Float32Array(src);

  const dst = new Float32Array(dstLen);
  const ratio = (srcLen - 1) / (dstLen - 1);

  for (let i = 0; i < dstLen; i++) {
    const srcIdx = i * ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, srcLen - 1);
    const frac = srcIdx - lo;
    dst[i] = src[lo]! * (1 - frac) + src[hi]! * frac;
  }

  return dst;
}

/**
 * Preprocess a raw 12-lead ECG signal for model inference.
 *
 * @param signal  Raw signal as array of 12 Float32Arrays (one per lead, in mV).
 *                Each lead can have any length (will be resampled to 5000).
 *                Lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
 * @param sex     Patient sex: 'M' or 'F' (default 'M')
 * @param age     Patient age in years (default 50)
 * @returns PreprocessedInput with signal and aux tensors
 */
export function preprocess(
  signal: Float32Array[],
  sex: 'M' | 'F' = 'M',
  age: number = 50,
): PreprocessedInput {
  if (signal.length !== N_LEADS) {
    throw new Error(`Expected ${N_LEADS} leads, got ${signal.length}`);
  }

  // Step 1: Resample each lead to SIGNAL_LEN if needed
  const resampled: Float32Array[] = new Array(N_LEADS);
  for (let i = 0; i < N_LEADS; i++) {
    resampled[i] = resampleLead(signal[i]!, SIGNAL_LEN);
  }

  // Step 2: Extract aux features BEFORE normalization (features need raw mV values)
  // Python: aux = extract_voltage_features(sig, sex=sex, age=age)
  // which internally calls extract_rr_features and concatenates [14 voltage + 4 RR]
  const voltageFeats = extractVoltageFeatures(resampled, sex, age); // 14 features
  const rrFeats = extractRRFeatures(resampled);                     // 4 features

  const aux = new Float32Array(N_AUX);
  aux.set(voltageFeats, 0);    // indices 0-13
  aux.set(rrFeats, 14);        // indices 14-17

  // Step 3: Normalize signal by /5.0 (amplitude-preserving normalization)
  // Python: sig_norm = (sig / 5.0).astype(np.float32)
  const signalTensor = new Float32Array(1 * N_LEADS * SIGNAL_LEN);
  for (let lead = 0; lead < N_LEADS; lead++) {
    const leadData = resampled[lead]!;
    const offset = lead * SIGNAL_LEN;
    for (let s = 0; s < SIGNAL_LEN; s++) {
      signalTensor[offset + s] = leadData[s]! / NORM_SCALE;
    }
  }

  return { signal: signalTensor, aux };
}
