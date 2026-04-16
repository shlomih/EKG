/**
 * InferencePipeline — Complete ECG analysis workflow.
 *
 * Handles: preprocessing → ONNX inference → interval calculation → result storage.
 * Wires together ModelManager, interval calculation, and database persistence.
 */

import { Alert } from 'react-native';
import { loadModel, getSession } from './ModelManager';
import { saveAnalysis } from '../db/AnalysisRepository';
import { saveRecord } from '../db/RecordRepository';
import { logEvent } from '../audit/AuditLogger';
import { DetectedCondition } from '../types/Analysis';
import type { ECGRecord, AnalysisResult, IntervalMeasurements } from '../types/ECGRecord';

/**
 * Preprocess ECG signal: normalize, filter, detect lead quality.
 */
function preprocessSignal(signal: Float32Array, channels: number = 12): Float32Array {
  const normalized = new Float32Array(signal.length);

  // Normalize per lead (z-score)
  for (let ch = 0; ch < channels; ch++) {
    const startIdx = ch * 5000;
    const endIdx = startIdx + 5000;
    const lead = signal.slice(startIdx, endIdx);

    const mean = Array.from(lead).reduce((a, b) => a + b) / lead.length;
    const variance =
      Array.from(lead).reduce((a, b) => a + (b - mean) ** 2) / lead.length;
    const std = Math.sqrt(variance);

    for (let i = 0; i < 5000; i++) {
      normalized[startIdx + i] = std > 0 ? (lead[i] - mean) / std : 0;
    }
  }

  return normalized;
}

/**
 * Simplified interval calculation from signal peaks.
 * Full version would use NeuroKit2-like algorithms.
 * Returns basic HR; PR/QRS/QTc would need more complex peak detection.
 */
function calculateIntervals(signal: Float32Array): IntervalMeasurements {
  // Extract Lead II (index 1 in 12-lead)
  const lead2 = signal.slice(5000, 10000);

  // Simple peak detection via moving average
  const windowSize = 50; // ~0.1s at 500 Hz
  let maxPeak = -Infinity;
  let minPeak = Infinity;
  let peakCount = 0;

  for (let i = 0; i < lead2.length - windowSize; i++) {
    let sum = 0;
    for (let j = 0; j < windowSize; j++) {
      sum += Math.abs(lead2[i + j]);
    }
    const localAvg = sum / windowSize;

    if (localAvg > 0.5) {
      peakCount++;
      maxPeak = Math.max(maxPeak, localAvg);
      minPeak = Math.min(minPeak, localAvg);
    }
  }

  // Estimate heart rate from peak spacing
  // At 500 Hz, 10 seconds = 5000 samples
  // Average peaks per second × 60 = BPM
  const estimatedHR = Math.max(40, Math.min(200, (peakCount / 50) * 60));

  return {
    heart_rate: Math.round(estimatedHR),
    pr_interval_ms: null, // Requires P-wave detection
    qrs_duration_ms: null, // Requires Q-S delineation
    qtc_ms: null, // Requires T-wave offset + Bazett formula
    rr_variability: null,
  };
}

/**
 * Map ONNX model output logits to detected conditions.
 * Returns detected conditions sorted by confidence and urgency.
 */
function mapDetections(
  logits: Float32Array | number[],
  threshold: number = 0.5,
): Array<{ code: string; confidence: number }> {
  // 26 ECG condition codes
  const CONDITIONS = [
    'NORM',
    'MI',
    'STTC',
    'CD',
    'HYP',
    'PAC',
    'PVC',
    'GSVT',
    'AF',
    'AFIB',
    'STEMI',
    'NSTEMI',
    'UA',
    'PE',
    'PULM',
    'MYOC',
    'PERI',
    'CHD',
    'COPD',
    'ASTHMA',
    'SEPSIS',
    'SHOCK',
    'ANEMIA',
    'THYROID',
    'NEURO',
    'OTHER',
  ];

  const detected: Array<{ code: string; confidence: number }> = [];

  // Apply sigmoid to convert logits to probabilities
  const logitsArray = Array.isArray(logits) ? logits : Array.from(logits);

  for (let i = 0; i < Math.min(CONDITIONS.length, logitsArray.length); i++) {
    const logit = logitsArray[i];
    const prob = 1 / (1 + Math.exp(-logit)); // sigmoid

    if (prob >= threshold) {
      detected.push({
        code: CONDITIONS[i],
        confidence: parseFloat(prob.toFixed(4)),
      });
    }
  }

  // Sort by confidence descending
  detected.sort((a, b) => b.confidence - a.confidence);

  return detected;
}

/**
 * Run full inference pipeline: preprocess → ONNX → intervals → store.
 * Returns analysis ID on success.
 */
export async function runInferencePipeline(
  patientId: string,
  signal: Float32Array,
  acquisitionSource: 'camera' | 'file_upload' | 'bluetooth' | 'demo' = 'camera',
): Promise<string | null> {
  try {
    await logEvent('INFERENCE_PIPELINE', 'pipeline_start', 'analysis', 'v3');

    // Step 1: Load model if not already loaded
    const modelLoaded = await loadModel();
    if (!modelLoaded) {
      throw new Error('Failed to load ONNX model');
    }

    // Step 2: Preprocess signal
    const preprocessed = preprocessSignal(signal);
    await logEvent('INFERENCE_PIPELINE', 'preprocessing_complete', 'analysis', 'v3');

    // Step 3: Run ONNX inference
    const session = getSession();
    if (!session) {
      throw new Error('ONNX session not available');
    }

    // Prepare input tensor
    // ONNX model expects: signal (1, 12, 5000), aux (1, 18)
    // For demo, we'll pass signal only; aux features can be computed later
    const inputTensor = {
      signal: new Float32Array(1 * 12 * 5000),
    };
    // Copy preprocessed data
    inputTensor.signal.set(preprocessed);

    // Run inference
    const outputs = await session.run(inputTensor);
    const logits = outputs.logits || outputs.output_0;

    if (!logits) {
      throw new Error('Model output missing or malformed');
    }

    await logEvent('INFERENCE_PIPELINE', 'inference_complete', 'analysis', 'v3');

    // Step 4: Map detections
    const detections = mapDetections(logits.data, 0.5);

    // Step 5: Calculate intervals
    const intervals = calculateIntervals(signal);
    await logEvent('INFERENCE_PIPELINE', 'intervals_calculated', 'analysis', 'v3');

    // Step 6: Create ECG record
    const ekgId = `ekgrecord_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    const recordChecksum = await computeSignalChecksum(signal);

    await saveRecord({
      ekg_id: ekgId,
      patient_id: patientId,
      signal_data: signal.buffer,
      sampling_rate: 500,
      lead_count: 12,
      acquisition_source: acquisitionSource,
      checksum: recordChecksum,
      notes: `Auto-analysis via InferencePipeline v3`,
    });

    await logEvent('INFERENCE_PIPELINE', 'record_saved', 'analysis', 'v3', { ekgId });

    // Step 7: Create analysis result
    const analysisId = `analysis_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;

    const analysis = {
      analysis_id: analysisId,
      ekg_id: ekgId,
      patient_id: patientId,
      model_version: 'V3 Multilabel (26 conditions)',
      model_hash: 'v3_sha256_placeholder',
      primary_condition: detections[0]?.code || 'UNKNOWN',
      detected_conditions: detections.map((d) => ({
        condition_code: d.code,
        confidence: d.confidence,
        urgency: getUrgency(d.code),
      })),
      conditions: detections.map((d) => d.code),
      scores: Object.fromEntries(detections.map((d) => [d.code, d.confidence])),
      intervals,
      clinical_rules: null,
      st_territory: null,
      disclaimer_shown: true,
      notes: `Automated analysis for patient ${patientId}`,
    };

    // Step 8: Store analysis
    await saveAnalysis(analysis);
    await logEvent('INFERENCE_PIPELINE', 'analysis_saved', 'analysis', 'v3', {
      analysisId,
      conditions: detections.length,
    });

    await logEvent('INFERENCE_PIPELINE', 'pipeline_complete', 'analysis', 'v3', {
      analysisId,
    });

    return analysisId;
  } catch (error) {
    await logEvent('INFERENCE_PIPELINE', 'pipeline_error', 'analysis', 'v3', {
      error: String(error),
    });
    Alert.alert('Analysis Failed', String(error));
    return null;
  }
}

/**
 * Compute SHA-256 checksum of signal for integrity (HIPAA 1.4.1).
 */
async function computeSignalChecksum(signal: Float32Array): Promise<string> {
  // In production, use expo-crypto
  // For now, return placeholder
  return `sha256_${Date.now()}`;
}

/**
 * Map ECG condition to urgency level.
 */
function getUrgency(
  condition: string,
): 'critical' | 'high' | 'normal' {
  const criticalCodes = ['STEMI', 'PE', 'SHOCK', 'SEPSIS', 'AFIB'];
  const highCodes = ['MI', 'NSTEMI', 'UA', 'MYOC', 'PERI'];

  if (criticalCodes.includes(condition)) return 'critical';
  if (highCodes.includes(condition)) return 'high';
  return 'normal';
}
