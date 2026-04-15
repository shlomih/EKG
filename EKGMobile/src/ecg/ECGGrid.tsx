/**
 * ECGGrid — draws the standard ECG background grid using Skia.
 *
 * Standard ECG paper:
 * - Small squares: 1mm (0.04s x 0.1mV)
 * - Large squares: 5mm (0.2s x 0.5mV)
 * - Paper speed: 25mm/s, gain: 10mm/mV
 */

import { Path, Group, Line } from '@shopify/react-native-skia';

interface ECGGridProps {
  width: number;
  height: number;
  /** Pixels per second (default: 25mm/s mapped to screen). */
  pxPerSec: number;
  /** Pixels per mV (default: 10mm/mV mapped to screen). */
  pxPerMv: number;
}

export default function ECGGrid({ width, height, pxPerSec, pxPerMv }: ECGGridProps) {
  // Large grid: 0.2s intervals horizontal, 0.5mV intervals vertical
  const largeH = pxPerSec * 0.2;
  const largeV = pxPerMv * 0.5;

  // Small grid: 0.04s intervals horizontal, 0.1mV intervals vertical
  const smallH = pxPerSec * 0.04;
  const smallV = pxPerMv * 0.1;

  const smallLines: { x1: number; y1: number; x2: number; y2: number }[] = [];
  const largeLines: { x1: number; y1: number; x2: number; y2: number }[] = [];

  // Vertical lines (time axis)
  for (let x = 0; x <= width; x += smallH) {
    const isLarge = Math.abs(x % largeH) < 0.5;
    (isLarge ? largeLines : smallLines).push({ x1: x, y1: 0, x2: x, y2: height });
  }

  // Horizontal lines (voltage axis)
  for (let y = 0; y <= height; y += smallV) {
    const isLarge = Math.abs(y % largeV) < 0.5;
    (isLarge ? largeLines : smallLines).push({ x1: 0, y1: y, x2: width, y2: y });
  }

  return (
    <Group>
      {/* Small grid lines */}
      {smallLines.map((l, i) => (
        <Line
          key={`s${i}`}
          p1={{ x: l.x1, y: l.y1 }}
          p2={{ x: l.x2, y: l.y2 }}
          color="rgba(0, 229, 176, 0.06)"
          strokeWidth={0.5}
        />
      ))}
      {/* Large grid lines */}
      {largeLines.map((l, i) => (
        <Line
          key={`l${i}`}
          p1={{ x: l.x1, y: l.y1 }}
          p2={{ x: l.x2, y: l.y2 }}
          color="rgba(0, 229, 176, 0.15)"
          strokeWidth={1}
        />
      ))}
    </Group>
  );
}
