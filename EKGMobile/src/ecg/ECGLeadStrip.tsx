/**
 * ECGLeadStrip — renders a single ECG lead waveform using Skia Path.
 *
 * Draws the signal as a continuous path on a Skia Canvas.
 * The signal values are in mV, mapped to pixels via pxPerMv.
 */

import { Path, Skia, Group, Text as SkiaText, useFont } from '@shopify/react-native-skia';
import { useMemo } from 'react';

interface ECGLeadStripProps {
  /** Signal samples in mV. */
  signal: Float32Array;
  /** Sampling rate in Hz (typically 500). */
  fs: number;
  /** Lead name (e.g., "I", "V1"). */
  leadName: string;
  /** Width of this strip in pixels. */
  width: number;
  /** Height of this strip in pixels. */
  height: number;
  /** Pixels per second. */
  pxPerSec: number;
  /** Pixels per millivolt. */
  pxPerMv: number;
  /** Start sample index (for time windowing). */
  startSample?: number;
  /** Number of samples to display. */
  sampleCount?: number;
  /** Signal color. */
  color?: string;
}

export default function ECGLeadStrip({
  signal,
  fs,
  leadName,
  width,
  height,
  pxPerSec,
  pxPerMv,
  startSample = 0,
  sampleCount,
  color = '#00E5B0',
}: ECGLeadStripProps) {
  const count = sampleCount ?? signal.length - startSample;
  const baselineY = height / 2; // vertical center = 0 mV

  const path = useMemo(() => {
    const p = Skia.Path.Make();
    const end = Math.min(startSample + count, signal.length);

    if (end <= startSample) return p;

    // First point
    const x0 = 0;
    const y0 = baselineY - (signal[startSample]! * pxPerMv);
    p.moveTo(x0, y0);

    // Draw path: each sample maps to x = (sampleIndex / fs) * pxPerSec
    for (let i = startSample + 1; i < end; i++) {
      const t = (i - startSample) / fs; // time in seconds from start
      const x = t * pxPerSec;
      const y = baselineY - (signal[i]! * pxPerMv); // negative because screen Y is inverted
      if (x > width) break;
      p.lineTo(x, y);
    }

    return p;
  }, [signal, fs, startSample, count, width, baselineY, pxPerSec, pxPerMv]);

  return (
    <Group>
      {/* Lead label */}
      <SkiaText
        x={4}
        y={14}
        text={leadName}
        color="#00E5B0"
        font={null}
      />
      {/* Signal trace */}
      <Path
        path={path}
        color={color}
        style="stroke"
        strokeWidth={1.2}
        strokeCap="round"
        strokeJoin="round"
      />
    </Group>
  );
}
