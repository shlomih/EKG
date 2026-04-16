/**
 * DemoData — synthetic ECG generation for testing without real hardware.
 *
 * Generates normal sinus rhythm (NSR) 12-lead ECG waveforms.
 * Useful for end-to-end testing of analysis pipeline.
 */

/**
 * Generate a synthetic normal sinus rhythm ECG signal.
 * Returns Float32Array of shape (12, 5000) at 500 Hz (10 seconds).
 */
export function generateNormalECG(): Float32Array {
  const CHANNELS = 12;
  const SAMPLES = 5000;
  const SAMPLING_RATE = 500; // Hz
  const DURATION = SAMPLES / SAMPLING_RATE; // 10 seconds

  const signal = new Float32Array(CHANNELS * SAMPLES);

  // Heart rate = 72 bpm = 1.2 Hz
  // P-wave: ~0.1s duration, ~0.15 mV amplitude
  // QRS: ~0.08s, ~1.5 mV
  // T-wave: ~0.2s, ~0.5 mV
  // One cardiac cycle ≈ 0.833s (at 72 bpm)

  const heartRate = 72; // bpm
  const cycleLength = (60 / heartRate) * SAMPLING_RATE; // samples per cycle

  // ECG wave morphologies (approximate sine/cosine envelopes)
  const createPWave = (t: number): number => {
    const pStart = 0.15 * cycleLength;
    const pDuration = 0.1 * cycleLength;
    if (t < pStart || t > pStart + pDuration) return 0;
    const pPos = (t - pStart) / pDuration;
    return 0.15 * Math.sin(Math.PI * pPos);
  };

  const createQRS = (t: number): number => {
    const qStart = 0.26 * cycleLength;
    const qDuration = 0.08 * cycleLength;
    if (t < qStart || t > qStart + qDuration) return 0;
    const qrsPos = (t - qStart) / qDuration;
    // Biphasic Q-R-S wave
    if (qrsPos < 0.3) return -0.2 * Math.sin(Math.PI * (qrsPos / 0.3));
    if (qrsPos < 0.7) return 1.5 * Math.sin(Math.PI * ((qrsPos - 0.3) / 0.4));
    return -0.5 * Math.sin(Math.PI * ((qrsPos - 0.7) / 0.3));
  };

  const createTWave = (t: number): number => {
    const tStart = 0.36 * cycleLength;
    const tDuration = 0.2 * cycleLength;
    if (t < tStart || t > tStart + tDuration) return 0;
    const tPos = (t - tStart) / tDuration;
    return 0.4 * Math.sin(Math.PI * tPos);
  };

  // Lead definitions: directions in cardiac coordinate system
  // Using standard ECG lead vectors
  const leadAngles: Record<number, { angle: number; amplitude: number }> = {
    0: { angle: 0, amplitude: 1.0 }, // I — lateral
    1: { angle: 60, amplitude: 0.5 }, // II — inferior
    2: { angle: -60, amplitude: -0.5 }, // III — inferior
    3: { angle: 90, amplitude: 0.8 }, // aVR — right
    4: { angle: 30, amplitude: 0.9 }, // aVL — lateral
    5: { angle: -90, amplitude: -0.7 }, // aVF — inferior
    6: { angle: 0, amplitude: -0.2 }, // V1 — right ventricle
    7: { angle: 0, amplitude: 0.1 }, // V2 — right/left septum
    8: { angle: 0, amplitude: 0.4 }, // V3 — septum
    9: { angle: 0, amplitude: 0.9 }, // V4 — apex
    10: { angle: -30, amplitude: 0.7 }, // V5 — left ventricle
    11: { angle: -60, amplitude: 0.5 }, // V6 — left ventricle
  };

  // Generate signal for each channel
  for (let ch = 0; ch < CHANNELS; ch++) {
    const leadAngle = (leadAngles[ch]?.angle ?? 0) * (Math.PI / 180);
    const leadAmp = leadAngles[ch]?.amplitude ?? 1.0;

    for (let s = 0; s < SAMPLES; s++) {
      // Modulate by lead direction (simulates ECG lead sensitivity)
      const leadFactor = Math.cos(leadAngle) * leadAmp;

      // Get time within current cardiac cycle
      const cyclePosition = (s % cycleLength) / cycleLength;
      const t = s;

      // Combine ECG waves with isoelectric baseline
      let waveform = createPWave(s % cycleLength);
      waveform += createQRS(s % cycleLength);
      waveform += createTWave(s % cycleLength);

      // Add slight baseline wander and noise
      const baselineWander = 0.05 * Math.sin((2 * Math.PI * s) / (10 * SAMPLING_RATE));
      const noise = 0.01 * (Math.random() - 0.5);

      // Apply lead factor and store
      signal[ch * SAMPLES + s] = leadFactor * (waveform + baselineWander + noise);
    }
  }

  return signal;
}

/**
 * Create a demo ECG record with metadata.
 */
export function createDemoRecord() {
  return {
    signal_data: generateNormalECG(),
    sampling_rate: 500,
    lead_count: 12,
    acquisition_source: 'demo' as const,
    notes: 'Synthetic NSR for testing',
  };
}
