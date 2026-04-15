/**
 * RRFeatures.ts — Port of extract_rr_features() from cnn_classifier.py:169-219
 *
 * Computes 4 RR-interval / rhythm features from Lead II.
 * These features capture the irregularity that characterizes AFIB
 * (irregular RR intervals + absent P waves).
 *
 * Returns 4-dim Float32Array (model aux indices 14-17):
 *   14: mean_rr_norm     mean RR in seconds, clamped to [0.3, 2.0]
 *   15: sdnn_norm        std of RR intervals / 0.2 s, clamped to [0, 1]
 *   16: rmssd_norm       root-mean-square successive differences / 0.2 s, clamped to [0, 1]
 *   17: irregularity     coefficient of variation = SDNN / meanRR, clamped to [0, 0.5]
 *                        Normal sinus ~0.02-0.05; AFIB typically 0.15-0.5
 */

/** Standard 12-lead index for Lead II. */
const LEAD_II = 1;

/** Sampling frequency (Hz). */
const FS = 500;

/** Fallback: regular 75 bpm rhythm with normal variability. */
const FALLBACK = new Float32Array([0.80, 0.03, 0.03, 0.04]);

/** Clamp value to [min, max]. */
function clamp(x: number, min: number, max: number): number {
  return x < min ? min : x > max ? max : x;
}

/**
 * Simple R-peak detection using adaptive threshold on Lead II.
 * Uses scipy-style find_peaks logic: height threshold + minimum distance.
 */
function findRPeaks(lead: Float32Array): number[] {
  // Find max amplitude
  let maxAmp = 0;
  for (let i = 0; i < lead.length; i++) {
    if (lead[i]! > maxAmp) maxAmp = lead[i]!;
  }

  if (maxAmp < 0.05) return []; // flat / unreadable lead

  const height = Math.max(0.15 * maxAmp, 0.05); // adaptive threshold
  const minDist = 40; // min 80 ms between peaks at 500 Hz

  // Find all samples above height
  const candidates: number[] = [];
  for (let i = 0; i < lead.length; i++) {
    if (lead[i]! >= height) candidates.push(i);
  }
  if (candidates.length === 0) return [];

  // Group into peaks separated by minDist, keeping local max in each group
  const peaks: number[] = [];
  let groupStart = 0;

  for (let i = 1; i <= candidates.length; i++) {
    // End of group when gap > minDist or end of array
    if (i === candidates.length || candidates[i]! - candidates[i - 1]! > minDist) {
      // Find local max in this group
      let bestIdx = candidates[groupStart]!;
      let bestVal = lead[bestIdx]!;
      for (let j = groupStart + 1; j < i; j++) {
        const idx = candidates[j]!;
        if (lead[idx]! > bestVal) {
          bestVal = lead[idx]!;
          bestIdx = idx;
        }
      }
      peaks.push(bestIdx);
      groupStart = i;
    }
  }

  return peaks;
}

/**
 * Extract 4 RR-interval features from a 12-lead ECG signal.
 *
 * @param signal - Array of 12 Float32Arrays (leads), each of length N, values in mV
 * @returns Float32Array of 4 features (model aux indices 14-17)
 */
export function extractRRFeatures(signal: Float32Array[]): Float32Array {
  const leadII = signal[LEAD_II];
  if (!leadII || leadII.length === 0) return new Float32Array(FALLBACK);

  const peaks = findRPeaks(leadII);
  if (peaks.length < 3) return new Float32Array(FALLBACK);

  // RR intervals in seconds
  const rrSec: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    rrSec.push((peaks[i]! - peaks[i - 1]!) / FS);
  }

  // Mean RR
  let sum = 0;
  for (const rr of rrSec) sum += rr;
  const meanRR = sum / rrSec.length;

  // SDNN (standard deviation of RR intervals)
  let sumSqDiff = 0;
  for (const rr of rrSec) sumSqDiff += (rr - meanRR) ** 2;
  const sdnn = Math.sqrt(sumSqDiff / rrSec.length);

  // RMSSD (root mean square of successive differences)
  let sumSuccDiffSq = 0;
  for (let i = 1; i < rrSec.length; i++) {
    const diff = rrSec[i]! - rrSec[i - 1]!;
    sumSuccDiffSq += diff * diff;
  }
  const rmssd = rrSec.length > 1
    ? Math.sqrt(sumSuccDiffSq / (rrSec.length - 1))
    : 0.0;

  // Irregularity (coefficient of variation)
  const irr = sdnn / (meanRR + 1e-8);

  return new Float32Array([
    clamp(meanRR, 0.3, 2.0),           // index 14: mean_rr_norm
    clamp(sdnn / 0.2, 0.0, 1.0),       // index 15: sdnn_norm
    clamp(rmssd / 0.2, 0.0, 1.0),      // index 16: rmssd_norm
    clamp(irr, 0.0, 0.5),              // index 17: irregularity
  ]);
}
