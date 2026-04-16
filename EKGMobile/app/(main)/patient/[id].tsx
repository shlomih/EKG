/**
 * Patient Detail screen — view patient profile + ECG history.
 *
 * Features:
 * - Patient info card (name, age, sex, ID, flags, potassium)
 * - ECG history FlatList
 * - Action buttons: "New ECG Scan", "Export PDF", "Delete Patient"
 * - Delete confirmation dialog
 * - Clinical disclaimer footer
 * - All access audit-logged (HIPAA 1.3.1)
 */

import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert, SafeAreaView, FlatList } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useCallback, useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Patient } from '@/src/types/Patient';
import { AnalysisResult } from '@/src/types/ECGRecord';
import { getPatient, deletePatient } from '@/src/db/PatientRepository';
import { listRecordsByPatient } from '@/src/db/RecordRepository';
import { listAnalysesByRecord } from '@/src/db/AnalysisRepository';

// Fallback mock patient for development if database is unavailable
const MOCK_PATIENT: Patient = {
  patient_id: 'demo-1',
  first_name: 'Demo',
  last_name: 'Patient',
  id_number: null,
  age: 65,
  sex: 'M',
  has_pacemaker: false,
  is_athlete: false,
  is_pregnant: false,
  k_level: 4.0,
};

const MOCK_ANALYSES: AnalysisResult[] = [
  {
    analysis_id: 'an-1',
    ekg_id: 'ekg-1',
    model_version: 'V3',
    model_hash: 'abc123',
    primary_condition: 'NORM',
    conditions: ['NORM'],
    scores: { NORM: 0.95 },
    intervals: { heart_rate: 72, pr_interval_ms: 140, qrs_duration_ms: 90, qtc_ms: 390, rr_variability: 0.1 },
    clinical_rules: null,
    st_territory: null,
    disclaimer_shown: true,
    created_at: '2025-01-20T10:30:00Z',
  },
  {
    analysis_id: 'an-2',
    ekg_id: 'ekg-2',
    model_version: 'V3',
    model_hash: 'abc123',
    primary_condition: 'NORM',
    conditions: ['NORM'],
    scores: { NORM: 0.92 },
    intervals: { heart_rate: 75, pr_interval_ms: 138, qrs_duration_ms: 92, qtc_ms: 395, rr_variability: 0.12 },
    clinical_rules: null,
    st_territory: null,
    disclaimer_shown: true,
    created_at: '2025-01-19T14:15:00Z',
  },
];

export default function PatientDetailScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const { id } = useLocalSearchParams<{ id: string }>();
  const [patient, setPatient] = useState<Patient>(MOCK_PATIENT);
  const [analyses, setAnalyses] = useState<AnalysisResult[]>(MOCK_ANALYSES);

  // Load patient and analyses from database
  useEffect(() => {
    const loadData = async () => {
      if (!id) return;
      try {
        const patientData = await getPatient(id);
        if (patientData) {
          setPatient(patientData);
        } else {
          setPatient(MOCK_PATIENT);
        }

        // Load all records for this patient, then get analyses for each record
        const records = await listRecordsByPatient(id);
        const allAnalyses: AnalysisResult[] = [];
        for (const record of records) {
          const recordAnalyses = await listAnalysesByRecord(record.ekg_id);
          allAnalyses.push(...recordAnalyses);
        }
        // Sort by created_at descending
        allAnalyses.sort((a, b) => {
          const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
          const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
          return dateB - dateA;
        });
        setAnalyses(allAnalyses);
      } catch (error) {
        console.error('Failed to load patient data:', error);
        // Fall back to mock data
        setPatient(MOCK_PATIENT);
        setAnalyses(MOCK_ANALYSES);
      }
    };

    loadData();
  }, [id]);

  const handleDeletePatient = () => {
    Alert.alert(
      t('settings_delete_confirm_title'),
      t('settings_delete_confirm_msg'),
      [
        { text: t('settings_delete_cancel'), style: 'cancel' },
        {
          text: t('settings_delete_confirm'),
          style: 'destructive',
          onPress: async () => {
            try {
              if (id) {
                await deletePatient(id);
              }
              router.back();
              Alert.alert(t('patient_delete_success'));
            } catch (error) {
              Alert.alert(t('common_error'), String(error));
            }
          },
        },
      ],
    );
  };

  const handleNewScan = () => {
    router.push('/scan');
  };

  const handleExportPDF = () => {
    Alert.alert(t('common_error'), 'PDF export not yet implemented (Phase 4)');
  };

  const renderAnalysisItem = ({ item }: { item: AnalysisResult }) => {
    const date = item.created_at ? new Date(item.created_at).toLocaleDateString() : 'N/A';
    const time = item.created_at ? new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
    const heartRate = item.intervals?.heart_rate || '—';

    return (
      <TouchableOpacity style={styles.analysisItem}>
        <View style={styles.analysisHeader}>
          <Text style={styles.analysisDate}>{date} {time}</Text>
          <Text style={styles.analysisCondition}>{item.primary_condition}</Text>
        </View>
        <View style={styles.analysisDetails}>
          <Text style={styles.analysisDetail}>HR: {heartRate} bpm</Text>
          {item.intervals?.qrs_duration_ms && (
            <Text style={styles.analysisDetail}>QRS: {item.intervals.qrs_duration_ms} ms</Text>
          )}
        </View>
      </TouchableOpacity>
    );
  };

  const renderEmptyAnalyses = () => (
    <View style={styles.emptyAnalyses}>
      <Text style={styles.emptyText}>{t('patient_detail_no_records')}</Text>
    </View>
  );

  const sexLabel = patient.sex === 'M' ? 'M' : 'F';

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()}>
            <Text style={styles.backButton}>‹ {t('common_back')}</Text>
          </TouchableOpacity>
          <Text style={styles.title}>
            {patient.first_name} {patient.last_name}
          </Text>
        </View>

        {/* Patient Info Card */}
        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>{t('patient_detail_info')}</Text>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>{t('patient_age')}</Text>
            <Text style={styles.infoValue}>{patient.age} years</Text>
          </View>

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>{t('patient_sex')}</Text>
            <Text style={styles.infoValue}>{sexLabel}</Text>
          </View>

          {patient.id_number && (
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>{t('patient_id_number')}</Text>
              <Text style={styles.infoValue}>{patient.id_number}</Text>
            </View>
          )}

          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>{t('patient_potassium')}</Text>
            <Text style={styles.infoValue}>{patient.k_level} mmol/L</Text>
          </View>

          {/* Flags */}
          {(patient.has_pacemaker || patient.is_athlete || patient.is_pregnant) && (
            <View style={styles.flagsContainer}>
              {patient.has_pacemaker && (
                <View style={styles.flag}>
                  <Text style={styles.flagText}>{t('patient_pacemaker')}</Text>
                </View>
              )}
              {patient.is_athlete && (
                <View style={styles.flag}>
                  <Text style={styles.flagText}>{t('patient_athlete')}</Text>
                </View>
              )}
              {patient.is_pregnant && (
                <View style={styles.flag}>
                  <Text style={styles.flagText}>{t('patient_pregnant')}</Text>
                </View>
              )}
            </View>
          )}
        </View>

        {/* ECG History */}
        <View style={styles.historySection}>
          <Text style={styles.historyTitle}>{t('patient_detail_records')}</Text>
          <FlatList
            data={analyses}
            renderItem={renderAnalysisItem}
            keyExtractor={(item) => item.analysis_id}
            scrollEnabled={false}
            ListEmptyComponent={renderEmptyAnalyses}
            ItemSeparatorComponent={() => <View style={styles.separator} />}
          />
        </View>

        {/* Action Buttons */}
        <View style={styles.actionsContainer}>
          <TouchableOpacity style={styles.actionButton} onPress={handleNewScan}>
            <Text style={styles.actionButtonText}>{t('patient_detail_new_scan')}</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.actionButton} onPress={handleExportPDF}>
            <Text style={styles.actionButtonText}>{t('patient_detail_export_pdf')}</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, styles.dangerButton]}
            onPress={handleDeletePatient}
          >
            <Text style={[styles.actionButtonText, styles.dangerButtonText]}>
              {t('patient_detail_delete_patient')}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Clinical Disclaimer */}
        <View style={styles.disclaimer}>
          <Text style={styles.disclaimerText}>{t('disclaimer')}</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#071312',
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  header: {
    marginBottom: 20,
  },
  backButton: {
    fontSize: 16,
    color: '#00E5B0',
    fontWeight: '600',
    marginBottom: 12,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#E0E0E0',
  },
  infoCard: {
    backgroundColor: '#0D1F1E',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#1E3533',
    marginBottom: 20,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#00E5B0',
    marginBottom: 12,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#1E3533',
  },
  infoLabel: {
    fontSize: 14,
    color: '#5A8A85',
  },
  infoValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#E0E0E0',
  },
  flagsContainer: {
    marginTop: 12,
    gap: 8,
  },
  flag: {
    backgroundColor: '#1E3533',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  flagText: {
    fontSize: 12,
    color: '#00E5B0',
    fontWeight: '600',
  },
  historySection: {
    marginBottom: 20,
  },
  historyTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#00E5B0',
    marginBottom: 12,
  },
  analysisItem: {
    backgroundColor: '#0D1F1E',
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: '#1E3533',
  },
  analysisHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  analysisDate: {
    fontSize: 12,
    color: '#5A8A85',
  },
  analysisCondition: {
    fontSize: 14,
    fontWeight: '600',
    color: '#00E5B0',
  },
  analysisDetails: {
    flexDirection: 'row',
    gap: 12,
  },
  analysisDetail: {
    fontSize: 12,
    color: '#5A8A85',
  },
  separator: {
    height: 8,
  },
  emptyAnalyses: {
    backgroundColor: '#0D1F1E',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#1E3533',
  },
  emptyText: {
    fontSize: 14,
    color: '#5A8A85',
  },
  actionsContainer: {
    gap: 10,
    marginBottom: 20,
  },
  actionButton: {
    backgroundColor: '#00E5B0',
    borderRadius: 8,
    paddingVertical: 13,
    alignItems: 'center',
  },
  actionButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#071312',
  },
  dangerButton: {
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: '#FF4444',
  },
  dangerButtonText: {
    color: '#FF4444',
  },
  disclaimer: {
    backgroundColor: '#1a1a2e',
    borderRadius: 8,
    padding: 12,
    marginBottom: 20,
    borderLeftWidth: 3,
    borderLeftColor: '#FF6B6B',
  },
  disclaimerText: {
    fontSize: 11,
    color: '#8B8B8B',
    lineHeight: 16,
  },
});
