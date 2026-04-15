/**
 * VoltageFeatures.ts — Port of extract_voltage_features() from cnn_classifier.py:222-292
 *
 * Computes a 14-dim voltage + demographic feature vector from raw 12-lead ECG signal (mV).
 *
 * Indices 0-10 (original):
 *   0: S(V1)              / 3.0   S-wave depth in V1 (LVH marker)
 *   1: max(R_V5, R_V6)   / 3.0   Tall R in lateral leads (LVH marker)
 *   2: Sokolow-Lyon value / 5.0   S(V1)+max(R_V5,R_V6); threshold 3.5 mV
 *   3: Sokolow met                1.0 if > 3.5 mV, else 0.0
 *   4: R(aVL)            / 2.0   R-wave in aVL (Cornell LVH)
 *   5: S(V3)             / 2.0   S-wave in V3 (Cornell LVH)
 *   6: Cornell value     / 4.0   R(aVL)+S(V3); threshold 2.8/2.0 mV
 *   7: R(V1)             / 2.0   Dominant R in V1 (RVH marker)
 *   8: sex_female                0.0 (M) / 1.0 (F)
 *   9: age_norm                  age / 80 [0,1]
 *  10: frontal QRS axis  / 180   deg
 *
 * Indices 11-13 (v10 morphology):
 *  11: T-wave strain score       LVH strain in V5/V6/aVL [0,1]
 *  12: QRS duration norm         QRS_ms / 200
 *  13: Cornell VDP norm          (Cornell x QRS_ms) / 2440
 */

/**
 * Standard 12-lead index mapping (PTB-XL WFDB order).
 * Signal array is [12][N] where each row is a lead in mV.
 */
const LEAD_IDX = {
  I: 0, II: 1, III: 2, AVR: 3, AVL: 4, AVF: 5,
  V1: 6, V2: 7, V3: 8, V4: 9, V5: 10, V6: 11,
} as const;

/** Clamp value to [min, max]. */
function clamp(x: number, min: number, max: number): number {
  return x < min ? min : x > max ? max : x;
}

/**
 * Get R-wave amplitude (max) and S-wave depth (abs(min)) for a lead.
 */
function rs(signal: Float32Array): [number, number] {
  let maxVal = -Infinity;
  let minVal = Infinity;
  for (let i = 0; i < signal.length; i++) {
    const v = signal[i]!;
    if (v > maxVal) maxVal = v;
    if (v < minVal) minVal = v;
  }
  return [maxVal, Math.abs(minVal)];
}

/**
 * Simple R-peak detection via threshold crossing on a single lead.
 * Returns sample indices of detected R-peaks.
 */
function detectRPeaksSimple(leadSig: Float32Array, fs: number): number[] {
  // Compute std
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < leadSig.length; i++) {
    const v = leadSig[i]!;
    sum += v;
    sumSq += v * v;
  }
  const mean = sum / leadSig.length;
  const variance = sumSq / leadSig.length - mean * mean;
  const std = Math.sqrt(Math.max(0, variance));

  if (std < 1e-6) return [];

  const threshold = 2.0 * std;
  const minDist = Math.floor(fs / 2);
  const win = Math.floor(fs / 10);

  // Find samples above threshold
  const above: number[] = [];
  for (let i = 0; i < leadSig.length; i++) {
    if (leadSig[i]! > threshold) above.push(i);
  }
  if (above.length === 0) return [];

  // Group into crossings (>= minDist apart)
  const crossings: number[] = [above[0]!];
  for (let i = 1; i < above.length; i++) {
    if (above[i]! - above[i - 1]! > minDist) {
      crossings.push(above[i]!);
    }
  }

  // Refine to local max within +/- win
  const refined: number[] = [];
  for (const c of crossings) {
    const start = Math.max(0, c - win);
    const end = Math.min(leadSig.length, c + win);
    let bestIdx = start;
    let bestVal = leadSig[start]!;
    for (let i = start + 1; i < end; i++) {
      if (leadSig[i]! > bestVal) {
        bestVal = leadSig[i]!;
        bestIdx = i;
      }
    }
    refined.push(bestIdx);
  }
  return refined;
}

/**
 * T-wave strain score: LVH strain pattern in lateral leads (V5, V6, aVL).
 * Returns score in [0, 1]. 1 = strongly inverted T waves.
 */
function tWaveStrainScore(signal: Float32Array[], fs: number): number {
  const leadII = signal[LEAD_IDX.II]!;
  const rPeaks = detectRPeaksSimple(leadII, fs);
  if (rPeaks.length < 1) return 0.0;

  const tStart = Math.floor(0.15 * fs); // 150 ms post-R
  const tEnd = Math.floor(0.38 * fs);   // 380 ms post-R

  const lateralLeads = [LEAD_IDX.V5, LEAD_IDX.V6, LEAD_IDX.AVL];
  const leadScores: number[] = [];

  for (const li of lateralLeads) {
    const lead = signal[li]!;
    // Remove DC
    let dcSum = 0;
    for (let i = 0; i < lead.length; i++) dcSum += lead[i]!;
    const dc = dcSum / lead.length;

    let leadMax = -Infinity;
    let leadMin = Infinity;
    for (let i = 0; i < lead.length; i++) {
      const v = lead[i]! - dc;
      if (v > leadMax) leadMax = v;
      if (v < leadMin) leadMin = v;
    }
    const leadRange = Math.max(leadMax - leadMin, 0.1);

    const beatT: number[] = [];
    for (const pk of rPeaks) {
      const s = pk + tStart;
      const e = pk + tEnd;
      if (e < lead.length) {
        let tSum = 0;
        let tCount = 0;
        for (let i = s; i < e; i++) {
          tSum += lead[i]! - dc;
          tCount++;
        }
        beatT.push(tSum / tCount);
      }
    }

    if (beatT.length > 0) {
      // Median of beat T-wave means
      beatT.sort((a, b) => a - b);
      const mid = Math.floor(beatT.length / 2);
      const medianT = beatT.length % 2 === 0
        ? (beatT[mid - 1]! + beatT[mid]!) / 2
        : beatT[mid]!;
      leadScores.push(Math.max(0.0, -medianT / leadRange));
    }
  }

  if (leadScores.length === 0) return 0.0;
  return clamp(Math.max(...leadScores), 0.0, 1.0);
}

/**
 * Estimate QRS duration in lead II, normalized by 200 ms.
 * Returns value in [0, 2]. Normal ~0.40-0.50, LVH/BBB ~0.50-0.60+.
 */
function qrsDurationNorm(signal: Float32Array[], fs: number): number {
  const lead = signal[LEAD_IDX.II]!;
  // Remove mean
  let sum = 0;
  for (let i = 0; i < lead.length; i++) sum += lead[i]!;
  const mean = sum / lead.length;

  const centered = new Float32Array(lead.length);
  for (let i = 0; i < lead.length; i++) centered[i] = lead[i]! - mean;

  const rPeaks = detectRPeaksSimple(centered, fs);
  if (rPeaks.length < 1) return 0.5; // default ~100 ms

  const win = Math.floor(0.08 * fs); // +/- 80 ms
  const durations: number[] = [];

  for (const pk of rPeaks) {
    const s = Math.max(0, pk - win);
    const e = Math.min(centered.length, pk + win);
    const seg = centered.subarray(s, e);
    if (seg.length < 4) continue;

    let peakAmp = 0;
    for (let i = 0; i < seg.length; i++) {
      const abs = Math.abs(seg[i]!);
      if (abs > peakAmp) peakAmp = abs;
    }
    if (peakAmp < 1e-6) continue;

    const aboveThreshold = 0.15 * peakAmp;
    let first = -1;
    let last = -1;
    for (let i = 0; i < seg.length; i++) {
      if (Math.abs(seg[i]!) > aboveThreshold) {
        if (first === -1) first = i;
        last = i;
      }
    }
    if (first >= 0 && last > first) {
      durations.push((last - first) / fs);
    }
  }

  if (durations.length === 0) return 0.5;

  // Median of durations
  durations.sort((a, b) => a - b);
  const mid = Math.floor(durations.length / 2);
  const median = durations.length % 2 === 0
    ? (durations[mid - 1]! + durations[mid]!) / 2
    : durations[mid]!;

  return clamp(median / 0.20, 0.0, 2.0);
}

/**
 * Extract 14-dim voltage + demographic features from raw 12-lead signal.
 *
 * @param signal - Array of 12 Float32Arrays, each of length N (samples), values in mV
 * @param sex - 'M' or 'F'
 * @param age - Patient age in years
 * @returns Float32Array of 14 features (indices 0-13)
 */
export function extractVoltageFeatures(
  signal: Float32Array[],
  sex: 'M' | 'F' = 'M',
  age: number = 50,
): Float32Array {
  // R and S wave amplitudes from key leads
  const [rI, sI] = rs(signal[LEAD_IDX.I]!);
  const [rAvf, sAvf] = rs(signal[LEAD_IDX.AVF]!);
  const [, sV1] = rs(signal[LEAD_IDX.V1]!);
  const [rV5] = rs(signal[LEAD_IDX.V5]!);
  const [rV6] = rs(signal[LEAD_IDX.V6]!);
  const [rAvl] = rs(signal[LEAD_IDX.AVL]!);
  const [, sV3] = rs(signal[LEAD_IDX.V3]!);
  const [rV1] = rs(signal[LEAD_IDX.V1]!);

  // Sokolow-Lyon: S(V1) + max(R(V5), R(V6))
  const sokolow = sV1 + Math.max(rV5, rV6);
  // Cornell: R(aVL) + S(V3)
  const cornell = rAvl + sV3;

  // Frontal QRS axis (degrees)
  const netI = rI - sI;
  const netAvf = rAvf - sAvf;
  const axisDeg = Math.atan2(netAvf, netI) * (180 / Math.PI);
  const axisNorm = clamp(axisDeg / 180.0, -1.0, 1.0);

  // Morphology features (v10)
  const nSamples = signal[0]!.length;
  const fsEst = Math.floor(nSamples / 10); // assume 10-second recording
  const tStrain = tWaveStrainScore(signal, fsEst);
  const qrsNorm = qrsDurationNorm(signal, fsEst);
  const qrsMs = qrsNorm * 200.0;
  const cvdpNorm = clamp((cornell * qrsMs) / 2440.0, 0.0, 2.0);

  return new Float32Array([
    clamp(sV1 / 3.0, 0, 2.0),              // 0: S(V1) normalized
    clamp(Math.max(rV5, rV6) / 3.0, 0, 2.0), // 1: R lateral normalized
    clamp(sokolow / 5.0, 0, 2.0),           // 2: Sokolow-Lyon normalized
    sokolow > 3.5 ? 1.0 : 0.0,              // 3: Sokolow criterion met
    clamp(rAvl / 2.0, 0, 2.0),              // 4: R(aVL) normalized
    clamp(sV3 / 2.0, 0, 2.0),               // 5: S(V3) normalized
    clamp(cornell / 4.0, 0, 2.0),           // 6: Cornell normalized
    clamp(rV1 / 2.0, 0, 2.0),               // 7: R(V1) normalized
    sex === 'F' ? 1.0 : 0.0,                // 8: sex_female
    clamp(age / 80.0, 0, 1),                // 9: age_norm
    axisNorm,                                // 10: frontal QRS axis
    tStrain,                                 // 11: T-wave strain score
    qrsNorm,                                 // 12: QRS duration norm
    cvdpNorm,                                // 13: Cornell VDP norm
  ]);
}
