/**
 * Scan screen — camera capture for paper ECG digitization.
 * Implementation pending (Phase 5).
 */

import { View, Text, StyleSheet } from 'react-native';

export default function ScanScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Scan ECG</Text>
      <Text style={styles.placeholder}>Camera capture — implementation pending (Phase 5)</Text>
      <Text style={styles.note}>
        HIPAA 5.3.6: Captured images will be processed in memory only.
        No images saved to Gallery/Photos.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#071312', padding: 20 },
  title: { fontSize: 24, fontWeight: '700', color: '#00E5B0', marginBottom: 16 },
  placeholder: { fontSize: 14, color: '#5A8A85', marginBottom: 8 },
  note: { fontSize: 12, color: '#3D6662', fontStyle: 'italic' },
});
