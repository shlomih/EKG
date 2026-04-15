/**
 * Settings screen — app configuration + HIPAA features.
 *
 * Includes:
 * - Audit log viewer (HIPAA 1.3.3)
 * - "Delete All My Data" (HIPAA 5.1.2)
 * - Inactivity timeout setting
 * - Language selection
 * - Model version info
 */

import { useState, useCallback } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert, Modal, FlatList } from 'react-native';
import { deleteAllData } from '../../src/db/PatientRepository';
import { getRecentEntries } from '../../src/audit/AuditLogger';
import { type AuditEntry } from '../../src/audit/AuditTypes';
import useAppStore from '../../src/store/useAppStore';
import i18n from '../../src/i18n/i18n';

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Espa\u00f1ol' },
  { code: 'fr', label: 'Fran\u00e7ais' },
];

const TIMEOUT_OPTIONS = [
  { ms: 60_000, label: '1 minute' },
  { ms: 300_000, label: '5 minutes' },
  { ms: 600_000, label: '10 minutes' },
];

export default function SettingsScreen() {
  const [auditLogVisible, setAuditLogVisible] = useState(false);
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const { language, setLanguage, inactivityTimeoutMs, setInactivityTimeout } = useAppStore();

  const handleDeleteAllData = () => {
    Alert.alert(
      'Delete All Data',
      'This will permanently delete all patient records, ECG data, and analysis results. ' +
      'The audit log will be preserved for compliance. This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete Everything',
          style: 'destructive',
          onPress: async () => {
            try {
              await deleteAllData();
              Alert.alert('Data Deleted', 'All patient data has been removed.');
            } catch {
              Alert.alert('Error', 'Failed to delete data. Please try again.');
            }
          },
        },
      ],
    );
  };

  const handleViewAuditLog = useCallback(() => {
    const entries = getRecentEntries(100);
    setAuditEntries(entries);
    setAuditLogVisible(true);
  }, []);

  const handleLanguageChange = () => {
    const currentIndex = LANGUAGES.findIndex(l => l.code === language);
    const nextIndex = (currentIndex + 1) % LANGUAGES.length;
    const next = LANGUAGES[nextIndex]!;
    setLanguage(next.code);
    i18n.changeLanguage(next.code);
  };

  const handleTimeoutChange = () => {
    const currentIndex = TIMEOUT_OPTIONS.findIndex(t => t.ms === inactivityTimeoutMs);
    const nextIndex = (currentIndex + 1) % TIMEOUT_OPTIONS.length;
    const next = TIMEOUT_OPTIONS[nextIndex]!;
    setInactivityTimeout(next.ms);
  };

  const currentLanguageLabel = LANGUAGES.find(l => l.code === language)?.label ?? 'English';
  const currentTimeoutLabel = TIMEOUT_OPTIONS.find(t => t.ms === inactivityTimeoutMs)?.label ?? '5 minutes';

  const renderAuditEntry = ({ item }: { item: AuditEntry }) => (
    <View style={styles.auditEntry}>
      <View style={styles.auditHeader}>
        <Text style={styles.auditEvent}>{item.event_type}</Text>
        <Text style={styles.auditTime}>{item.timestamp}</Text>
      </View>
      <Text style={styles.auditAction}>{item.action}</Text>
      {item.resource_type && (
        <Text style={styles.auditResource}>{item.resource_type}: {item.resource_id}</Text>
      )}
    </View>
  );

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Settings</Text>

      {/* Security Section */}
      <Text style={styles.sectionTitle}>Security</Text>

      <TouchableOpacity style={styles.settingRow} onPress={handleViewAuditLog}>
        <Text style={styles.settingLabel}>View Audit Log</Text>
        <Text style={styles.settingHint}>HIPAA 1.3.3</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.settingRow} onPress={handleTimeoutChange}>
        <Text style={styles.settingLabel}>Auto-Lock Timeout</Text>
        <Text style={styles.settingValue}>{currentTimeoutLabel}</Text>
      </TouchableOpacity>

      <View style={styles.settingRow}>
        <Text style={styles.settingLabel}>Screen Protection</Text>
        <Text style={styles.settingValue}>Active</Text>
      </View>

      {/* Data Section */}
      <Text style={styles.sectionTitle}>Data</Text>

      <TouchableOpacity style={styles.settingRow} onPress={handleLanguageChange}>
        <Text style={styles.settingLabel}>Language</Text>
        <Text style={styles.settingValue}>{currentLanguageLabel}</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.settingRow, styles.dangerRow]}
        onPress={handleDeleteAllData}
      >
        <Text style={styles.dangerLabel}>Delete All My Data</Text>
        <Text style={styles.settingHint}>HIPAA 5.1.2</Text>
      </TouchableOpacity>

      {/* About Section */}
      <Text style={styles.sectionTitle}>About</Text>

      <View style={styles.settingRow}>
        <Text style={styles.settingLabel}>Model Version</Text>
        <Text style={styles.settingValue}>V3 (26 conditions)</Text>
      </View>

      <View style={styles.settingRow}>
        <Text style={styles.settingLabel}>Inference</Text>
        <Text style={styles.settingValue}>On-device (ONNX Runtime)</Text>
      </View>

      {/* Disclaimer */}
      <View style={styles.disclaimer}>
        <Text style={styles.disclaimerText}>
          For educational purposes only. This is not a medical diagnosis.
          Not FDA-cleared. Always consult a qualified healthcare professional.
        </Text>
      </View>

      {/* Audit Log Modal */}
      <Modal
        visible={auditLogVisible}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setAuditLogVisible(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Audit Log</Text>
            <TouchableOpacity onPress={() => setAuditLogVisible(false)}>
              <Text style={styles.modalClose}>Close</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.modalSubtitle}>{auditEntries.length} entries</Text>
          <FlatList
            data={auditEntries}
            keyExtractor={(item, index) => `${item.id ?? index}`}
            renderItem={renderAuditEntry}
            ListEmptyComponent={
              <Text style={styles.emptyText}>No audit entries yet.</Text>
            }
          />
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#071312' },
  content: { padding: 20 },
  title: { fontSize: 24, fontWeight: '700', color: '#00E5B0', marginBottom: 24 },
  sectionTitle: { fontSize: 14, fontWeight: '600', color: '#5A8A85', marginTop: 16, marginBottom: 8, textTransform: 'uppercase' },
  settingRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    backgroundColor: '#0D1F1E', borderRadius: 8, padding: 14, marginBottom: 8,
    borderWidth: 1, borderColor: '#1E3533',
  },
  settingLabel: { fontSize: 15, color: '#E0E0E0' },
  settingValue: { fontSize: 14, color: '#5A8A85' },
  settingHint: { fontSize: 11, color: '#3D6662' },
  dangerRow: { borderColor: '#FF4444' },
  dangerLabel: { fontSize: 15, color: '#FF4444', fontWeight: '600' },
  disclaimer: {
    backgroundColor: '#1a1a2e', borderRadius: 8, padding: 12, marginTop: 24,
    borderLeftWidth: 3, borderLeftColor: '#FF6B6B',
  },
  disclaimerText: { fontSize: 11, color: '#8B8B8B', lineHeight: 16 },
  modalContainer: { flex: 1, backgroundColor: '#071312', padding: 20 },
  modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  modalTitle: { fontSize: 22, fontWeight: '700', color: '#00E5B0' },
  modalClose: { fontSize: 16, color: '#00E5B0' },
  modalSubtitle: { fontSize: 12, color: '#5A8A85', marginBottom: 16 },
  auditEntry: {
    backgroundColor: '#0D1F1E', borderRadius: 8, padding: 12, marginBottom: 8,
    borderWidth: 1, borderColor: '#1E3533',
  },
  auditHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 4 },
  auditEvent: { fontSize: 13, fontWeight: '600', color: '#00E5B0' },
  auditTime: { fontSize: 11, color: '#3D6662' },
  auditAction: { fontSize: 13, color: '#E0E0E0' },
  auditResource: { fontSize: 11, color: '#5A8A85', marginTop: 2 },
  emptyText: { fontSize: 14, color: '#5A8A85', textAlign: 'center', marginTop: 40 },
});
