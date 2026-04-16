/**
 * Patient List screen — displays all patients.
 *
 * Features:
 * - FlatList of patients with name, age, sex, last updated
 * - Pull-to-refresh
 * - Tap patient to navigate to detail screen
 * - "Add Patient" floating button
 * - Empty state message
 * - Mock data fallback for development
 */

import { View, Text, StyleSheet, FlatList, RefreshControl, TouchableOpacity, SafeAreaView } from 'react-native';
import { useRouter } from 'expo-router';
import { useCallback, useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Patient } from '@/src/types/Patient';
import { listPatients } from '@/src/db/PatientRepository';

// Fallback mock data for development if database is unavailable
const MOCK_PATIENTS: Patient[] = [
  {
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
  },
  {
    patient_id: 'demo-2',
    first_name: 'Jane',
    last_name: 'Doe',
    id_number: '123456',
    age: 54,
    sex: 'F',
    has_pacemaker: true,
    is_athlete: false,
    is_pregnant: false,
    k_level: 3.8,
  },
];

export default function PatientsScreen() {
  const router = useRouter();
  const { t } = useTranslation();
  const [patients, setPatients] = useState<Patient[]>(MOCK_PATIENTS);
  const [refreshing, setRefreshing] = useState(false);

  // Load patients from database
  useEffect(() => {
    const loadPatients = async () => {
      try {
        const patientsData = await listPatients();
        setPatients(patientsData);
      } catch (error) {
        console.error('Failed to load patients:', error);
        // Fall back to mock data
        setPatients(MOCK_PATIENTS);
      }
    };

    loadPatients();
  }, []);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const patientsData = await listPatients();
      setPatients(patientsData);
    } catch (error) {
      console.error('Failed to refresh patients:', error);
    } finally {
      setRefreshing(false);
    }
  }, []);

  const handlePatientPress = (patientId: string) => {
    router.push(`/patient/${patientId}`);
  };

  const handleAddPatient = () => {
    router.push('/patient/new');
  };

  const renderPatientItem = ({ item }: { item: Patient }) => {
    const sexLabel = item.sex === 'M' ? 'M' : 'F';
    const updatedDate = item.updated_at ? new Date(item.updated_at).toLocaleDateString() : 'N/A';

    return (
      <TouchableOpacity
        style={styles.patientItem}
        onPress={() => handlePatientPress(item.patient_id)}
      >
        <View style={styles.patientInfo}>
          <Text style={styles.patientName}>
            {item.first_name} {item.last_name}
          </Text>
          <Text style={styles.patientMeta}>
            {t('patient_item_age_sex', { age: item.age, sex: sexLabel })}
          </Text>
          <Text style={styles.patientUpdated}>
            {t('patient_item_updated', { date: updatedDate })}
          </Text>
        </View>
        <Text style={styles.chevron}>›</Text>
      </TouchableOpacity>
    );
  };

  const renderEmpty = () => (
    <View style={styles.emptyContainer}>
      <Text style={styles.emptyIcon}>--</Text>
      <Text style={styles.emptyTitle}>{t('patient_no_patients')}</Text>
      <TouchableOpacity style={styles.emptyButton} onPress={handleAddPatient}>
        <Text style={styles.emptyButtonText}>{t('patient_add_new')}</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>{t('patient_list_title')}</Text>
      </View>

      <FlatList
        data={patients}
        renderItem={renderPatientItem}
        keyExtractor={(item) => item.patient_id}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor="#00E5B0"
            progressBackgroundColor="#0D1F1E"
          />
        }
        ListEmptyComponent={renderEmpty}
        scrollEnabled={true}
      />

      {/* Add Patient Floating Button */}
      <TouchableOpacity
        style={styles.fab}
        onPress={handleAddPatient}
      >
        <Text style={styles.fabText}>+</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#071312',
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 16,
    backgroundColor: '#071312',
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#00E5B0',
  },
  listContent: {
    paddingHorizontal: 20,
    paddingVertical: 8,
  },
  patientItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#0D1F1E',
    borderRadius: 12,
    padding: 14,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#1E3533',
  },
  patientInfo: {
    flex: 1,
  },
  patientName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#E0E0E0',
    marginBottom: 4,
  },
  patientMeta: {
    fontSize: 13,
    color: '#5A8A85',
    marginBottom: 2,
  },
  patientUpdated: {
    fontSize: 12,
    color: '#3D6662',
  },
  chevron: {
    fontSize: 20,
    color: '#00E5B0',
    marginLeft: 8,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 60,
  },
  emptyIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#5A8A85',
    marginBottom: 20,
    textAlign: 'center',
  },
  emptyButton: {
    backgroundColor: '#00E5B0',
    borderRadius: 8,
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  emptyButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#071312',
  },
  fab: {
    position: 'absolute',
    bottom: 24,
    right: 20,
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#00E5B0',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  fabText: {
    fontSize: 28,
    fontWeight: '700',
    color: '#071312',
  },
});
