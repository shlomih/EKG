/**
 * ECG12LeadView — standard 4x3 clinical ECG display + Lead II rhythm strip.
 *
 * Layout matches clinical standard:
 *   Row 0: I,   aVR, V1, V4    (each shows 2.5s window)
 *   Row 1: II,  aVL, V2, V5
 *   Row 2: III, aVF, V3, V6
 *   Row 3: Lead II rhythm strip (full 10s)
 *
 * Each column shows the same time window:
 *   Col 0: 0–2.5s, Col 1: 2.5–5s, Col 2: 5–7.5s, Col 3: 7.5–10s
 *
 * Uses @shopify/react-native-skia for GPU-accelerated rendering.
 */

import { View, Text, StyleSheet, useWindowDimensions } from 'react-native';
import { Canvas } from '@shopify/react-native-skia';
import ECGGrid from './ECGGrid';
import ECGLeadStrip from './ECGLeadStrip';

/** Standard clinical 4x3 grid layout. */
const TWELVE_LEAD_GRID = [
  ['I', 'aVR', 'V1', 'V4'],
  ['II', 'aVL', 'V2', 'V5'],
  ['III', 'aVF', 'V3', 'V6'],
] as const;

/** Lead name to signal array index (PTB-XL WFDB order). */
const LEAD_TO_INDEX: Record<string, number> = {
  I: 0, II: 1, III: 2, aVR: 3, aVL: 4, aVF: 5,
  V1: 6, V2: 7, V3: 8, V4: 9, V5: 10, V6: 11,
};

interface ECG12LeadViewProps {
  /** 12-lead signal: array of 12 Float32Arrays, each of length N samples. */
  signal: Float32Array[];
  /** Sampling rate in Hz (default 500). */
  fs?: number;
  /** Duration per cell in seconds (default 2.5). */
  cellDuration?: number;
}

export default function ECG12LeadView({
  signal,
  fs = 500,
  cellDuration = 2.5,
}: ECG12LeadViewProps) {
  const { width: screenWidth } = useWindowDimensions();
  const padding = 8;
  const totalWidth = screenWidth - padding * 2;

  // Grid cell dimensions
  const cols = 4;
  const rows = 3; // grid rows (rhythm strip is separate)
  const cellWidth = totalWidth / cols;
  const cellHeight = 60; // pixels per cell
  const rhythmHeight = 60; // rhythm strip height
  const totalHeight = cellHeight * rows + rhythmHeight + 20; // +20 for labels

  // Scale: pixels per second and per millivolt
  const pxPerSec = cellWidth / cellDuration;
  const pxPerMv = cellHeight / 3; // ~3mV range per cell

  const samplesPerCell = Math.floor(fs * cellDuration);

  if (!signal || signal.length !== 12) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>No 12-lead signal data available.</Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, { paddingHorizontal: padding }]}>
      {/* 4x3 Grid */}
      {TWELVE_LEAD_GRID.map((rowLeads, rowIdx) => (
        <View key={rowIdx} style={styles.gridRow}>
          {rowLeads.map((leadName, colIdx) => {
            const leadIdx = LEAD_TO_INDEX[leadName] ?? 0;
            const leadSignal = signal[leadIdx]!;
            const startSample = colIdx * samplesPerCell;

            return (
              <View key={leadName} style={[styles.cell, { width: cellWidth, height: cellHeight }]}>
                <Text style={styles.leadLabel}>{leadName}</Text>
                <Canvas style={{ width: cellWidth, height: cellHeight }}>
                  <ECGGrid
                    width={cellWidth}
                    height={cellHeight}
                    pxPerSec={pxPerSec}
                    pxPerMv={pxPerMv}
                  />
                  <ECGLeadStrip
                    signal={leadSignal}
                    fs={fs}
                    leadName=""
                    width={cellWidth}
                    height={cellHeight}
                    pxPerSec={pxPerSec}
                    pxPerMv={pxPerMv}
                    startSample={startSample}
                    sampleCount={samplesPerCell}
                  />
                </Canvas>
              </View>
            );
          })}
        </View>
      ))}

      {/* Lead II Rhythm Strip (full 10s) */}
      <View style={styles.rhythmContainer}>
        <Text style={styles.leadLabel}>II (rhythm strip)</Text>
        <Canvas style={{ width: totalWidth, height: rhythmHeight }}>
          <ECGGrid
            width={totalWidth}
            height={rhythmHeight}
            pxPerSec={totalWidth / 10} // 10 seconds across full width
            pxPerMv={rhythmHeight / 3}
          />
          <ECGLeadStrip
            signal={signal[LEAD_TO_INDEX['II']!]!}
            fs={fs}
            leadName=""
            width={totalWidth}
            height={rhythmHeight}
            pxPerSec={totalWidth / 10}
            pxPerMv={rhythmHeight / 3}
            startSample={0}
            sampleCount={Math.min(signal[1]!.length, fs * 10)}
          />
        </Canvas>
      </View>

      {/* Scale reference */}
      <Text style={styles.scaleText}>
        25 mm/s &middot; 10 mm/mV
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#071312',
  },
  gridRow: {
    flexDirection: 'row',
  },
  cell: {
    borderWidth: 0.5,
    borderColor: '#1E3533',
    position: 'relative',
  },
  leadLabel: {
    position: 'absolute',
    top: 2,
    left: 4,
    fontSize: 10,
    fontWeight: '600',
    color: '#00E5B0',
    zIndex: 1,
  },
  rhythmContainer: {
    borderWidth: 0.5,
    borderColor: '#1E3533',
    marginTop: 4,
    position: 'relative',
  },
  errorText: {
    fontSize: 14,
    color: '#FF6B6B',
    textAlign: 'center',
    padding: 20,
  },
  scaleText: {
    fontSize: 10,
    color: '#3D6662',
    textAlign: 'right',
    marginTop: 4,
  },
});
