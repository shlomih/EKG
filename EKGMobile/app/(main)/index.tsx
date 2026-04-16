/**
 * Dashboard — home screen after authentication.
 *
 * Shows: recent analyses, quick actions, model version info, model load status.
 * Clinical disclaimer always visible (HIPAA 5.3.5).
 */

import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import useAppStore from '@/src/store/useAppStore';

const MODEL_AUROC = '0.990';
const MODEL_VERSION = 'V3.2b Multilabel (26 conditions)';

export default function DashboardScreen() {
  const router = useRouter();
  const { modelLoaded, modelVersion } = useAppStore();

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>EKG Intelligence</Text>
      <Text style={styles.subtitle}>26-Condition ECG Analysis</Text>

      {/* Quick Actions */}
      <View style={styles.actionsRow}>
        <TouchableOpacity
          style={styles.actionCard}
          onPress={() => router.push('/scan')}
        >
          <Text style={styles.actionIcon}>📷</Text>
          <Text style={styles.actionLabel}>Scan ECG</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionCard}
          onPress={() => router.push('/scan?demo=true')}
        >
          <Text style={styles.actionIcon}>🧪</Text>
          <Text style={styles.actionLabel}>Demo</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionCard}
          onPress={() => router.push('/patients')}
        >
          <Text style={styles.actionIcon}>👤</Text>
          <Text style={styles.actionLabel}>Patients</Text>
        </TouchableOpacity>
      </View>

      {/* Model Info */}
      <View style={styles.infoCard}>
        <Text style={styles.infoTitle}>Model</Text>
        <Text style={styles.infoText}>{MODEL_VERSION}</Text>
        <Text style={styles.infoText}>ECGNetJoint — 1.7M parameters</Text>
        <Text style={styles.infoText}>AUROC: {MODEL_AUROC}</Text>
        <Text style={styles.infoText}>On-device inference (ONNX Runtime)</Text>
        <View style={styles.statusRow}>
          <Text style={[styles.statusBadge, modelLoaded ? styles.statusLoaded : styles.statusPending]}>
            {modelLoaded ? '✓ Loaded' : '○ Loading'}
          </Text>
        </View>
      </View>

      {/* Clinical Disclaimer — HIPAA 5.3.5 / FDA 7.1 */}
      <View style={styles.disclaimer}>
        <Text style={styles.disclaimerText}>
          For educational purposes only. This is not a medical diagnosis.
          Not FDA-cleared. Always consult a qualified healthcare professional.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#071312',
  },
  content: {
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#00E5B0',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#5A8A85',
    marginBottom: 24,
  },
  actionsRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 24,
    flexWrap: 'wrap',
  },
  actionCard: {
    flex: 1,
    minWidth: 100,
    backgroundColor: '#0D1F1E',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#1E3533',
  },
  actionIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  actionLabel: {
    fontSize: 14,
    color: '#00E5B0',
    fontWeight: '600',
  },
  infoCard: {
    backgroundColor: '#0D1F1E',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#1E3533',
    marginBottom: 24,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#00E5B0',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 13,
    color: '#5A8A85',
    marginBottom: 2,
  },
  statusRow: {
    marginTop: 12,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#1E3533',
  },
  statusBadge: {
    fontSize: 12,
    fontWeight: '600',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  statusLoaded: {
    color: '#00E5B0',
    backgroundColor: '#00E5B020',
  },
  statusPending: {
    color: '#FFD700',
    backgroundColor: '#FFD70020',
  },
  disclaimer: {
    backgroundColor: '#1a1a2e',
    borderRadius: 8,
    padding: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#FF6B6B',
  },
  disclaimerText: {
    fontSize: 11,
    color: '#8B8B8B',
    lineHeight: 16,
  },
});
