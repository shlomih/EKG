/**
 * Add Patient form screen — create a new patient record.
 *
 * Features:
 * - Text inputs for name, ID, age, potassium
 * - Sex selector (Male/Female buttons)
 * - Toggles for pacemaker, athlete, pregnancy (pregnancy only when sex=F)
 * - Form validation (first + last name required)
 * - Save to database (TODO: connect to PatientRepository)
 * - All access audit-logged
 */

import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Switch,
  ScrollView,
  StyleSheet,
  SafeAreaView,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { savePatient } from '@/src/db/PatientRepository';

export default function NewPatientScreen() {
  const router = useRouter();
  const { t } = useTranslation();

  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [idNumber, setIdNumber] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState<'M' | 'F'>('M');
  const [kLevel, setKLevel] = useState('');
  const [hasPacemaker, setHasPacemaker] = useState(false);
  const [isAthlete, setIsAthlete] = useState(false);
  const [isPregnant, setIsPregnant] = useState(false);

  const handleSave = async () => {
    // Validation
    if (!firstName.trim() || !lastName.trim()) {
      Alert.alert(t('common_error'), 'First name and last name are required.');
      return;
    }

    try {
      await savePatient({
        first_name: firstName.trim(),
        last_name: lastName.trim(),
        id_number: idNumber.trim() || null,
        age: age ? parseInt(age) : null,
        sex,
        k_level: kLevel ? parseFloat(kLevel) : null,
        has_pacemaker: hasPacemaker,
        is_athlete: isAthlete,
        is_pregnant: sex === 'F' ? isPregnant : false,
      });

      Alert.alert(t('patient_save_success'));
      router.back();
    } catch (error) {
      Alert.alert(t('common_error'), String(error));
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Text style={styles.backButton}>‹ {t('common_back')}</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{t('patient_add_new')}</Text>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* First Name */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>{t('patient_first_name')}</Text>
          <TextInput
            style={styles.input}
            placeholder={t('patient_first_name_placeholder')}
            placeholderTextColor="#3D6662"
            value={firstName}
            onChangeText={setFirstName}
          />
        </View>

        {/* Last Name */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>{t('patient_last_name')}</Text>
          <TextInput
            style={styles.input}
            placeholder={t('patient_last_name_placeholder')}
            placeholderTextColor="#3D6662"
            value={lastName}
            onChangeText={setLastName}
          />
        </View>

        {/* Patient ID (optional) */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>{t('patient_id_number')}</Text>
          <TextInput
            style={styles.input}
            placeholder={t('patient_id_number_placeholder')}
            placeholderTextColor="#3D6662"
            value={idNumber}
            onChangeText={setIdNumber}
          />
        </View>

        {/* Age */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>{t('patient_age')}</Text>
          <TextInput
            style={styles.input}
            placeholder="65"
            placeholderTextColor="#3D6662"
            keyboardType="numeric"
            value={age}
            onChangeText={setAge}
          />
        </View>

        {/* Sex Selector */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>{t('patient_sex')}</Text>
          <View style={styles.sexSelectorContainer}>
            <TouchableOpacity
              style={[styles.sexButton, sex === 'M' && styles.sexButtonActive]}
              onPress={() => setSex('M')}
            >
              <Text
                style={[
                  styles.sexButtonText,
                  sex === 'M' && styles.sexButtonTextActive,
                ]}
              >
                {t('patient_sex_male')}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.sexButton, sex === 'F' && styles.sexButtonActive]}
              onPress={() => setSex('F')}
            >
              <Text
                style={[
                  styles.sexButtonText,
                  sex === 'F' && styles.sexButtonTextActive,
                ]}
              >
                {t('patient_sex_female')}
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Potassium */}
        <View style={styles.formGroup}>
          <Text style={styles.label}>{t('patient_potassium')}</Text>
          <Text style={styles.hint}>{t('patient_potassium_help')}</Text>
          <TextInput
            style={styles.input}
            placeholder="4.0"
            placeholderTextColor="#3D6662"
            keyboardType="decimal-pad"
            value={kLevel}
            onChangeText={setKLevel}
          />
        </View>

        {/* Pacemaker Toggle */}
        <View style={styles.toggleGroup}>
          <Text style={styles.label}>{t('patient_pacemaker')}</Text>
          <Switch
            value={hasPacemaker}
            onValueChange={setHasPacemaker}
            trackColor={{ false: '#1E3533', true: '#00E5B0' }}
            thumbColor={hasPacemaker ? '#071312' : '#5A8A85'}
          />
        </View>

        {/* Athlete Toggle */}
        <View style={styles.toggleGroup}>
          <Text style={styles.label}>{t('patient_athlete')}</Text>
          <Switch
            value={isAthlete}
            onValueChange={setIsAthlete}
            trackColor={{ false: '#1E3533', true: '#00E5B0' }}
            thumbColor={isAthlete ? '#071312' : '#5A8A85'}
          />
        </View>

        {/* Pregnancy Toggle (only if female) */}
        {sex === 'F' && (
          <View style={styles.toggleGroup}>
            <Text style={styles.label}>{t('patient_pregnant')}</Text>
            <Switch
              value={isPregnant}
              onValueChange={setIsPregnant}
              trackColor={{ false: '#1E3533', true: '#00E5B0' }}
              thumbColor={isPregnant ? '#071312' : '#5A8A85'}
            />
          </View>
        )}

        {/* Save Button */}
        <TouchableOpacity style={styles.saveButton} onPress={handleSave}>
          <Text style={styles.saveButtonText}>{t('patient_save')}</Text>
        </TouchableOpacity>

        <View style={styles.spacer} />
      </ScrollView>
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
  scrollContent: {
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  formGroup: {
    marginBottom: 20,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#5A8A85',
    marginBottom: 6,
  },
  hint: {
    fontSize: 12,
    color: '#3D6662',
    marginBottom: 4,
  },
  input: {
    backgroundColor: '#0D1F1E',
    borderWidth: 1,
    borderColor: '#1E3533',
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: '#E0E0E0',
    fontSize: 15,
  },
  sexSelectorContainer: {
    flexDirection: 'row',
    gap: 12,
  },
  sexButton: {
    flex: 1,
    backgroundColor: '#0D1F1E',
    borderWidth: 1,
    borderColor: '#1E3533',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
  },
  sexButtonActive: {
    backgroundColor: '#00E5B0',
    borderColor: '#00E5B0',
  },
  sexButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#5A8A85',
  },
  sexButtonTextActive: {
    color: '#071312',
  },
  toggleGroup: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#0D1F1E',
    borderWidth: 1,
    borderColor: '#1E3533',
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
    marginBottom: 12,
  },
  saveButton: {
    backgroundColor: '#00E5B0',
    borderRadius: 8,
    paddingVertical: 15,
    alignItems: 'center',
    marginTop: 12,
  },
  saveButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#071312',
  },
  spacer: {
    height: 20,
  },
});
