/**
 * Scan screen — camera capture for paper ECG digitization.
 * Supports demo mode with synthetic ECG data and full inference pipeline.
 */

import { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { createDemoRecord } from '@/src/ml/DemoData';
import { runInferencePipeline } from '@/src/ml/InferencePipeline';
import useAppStore from '@/src/store/useAppStore';

export default function ScanScreen() {
  const router = useRouter();
  const { demo } = useLocalSearchParams<{ demo?: string }>();
  const { currentPatient } = useAppStore();
  const isDemo = demo === 'true';
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleUseDemoECG = async () => {
    if (!currentPatient) {
      Alert.alert('No Patient Selected', 'Please select a patient first');
      router.push('/patients');
      return;
    }

    setIsAnalyzing(true);
    try {
      // Generate synthetic ECG
      const demoRecord = createDemoRecord();

      // Run full inference pipeline
      const analysisId = await runInferencePipeline(
        currentPatient.id,
        demoRecord.signal_data,
        'demo',
      );

      if (analysisId) {
        // Navigate to analysis results
        router.push(`/analyze?analysisId=${analysisId}`);
      } else {
        Alert.alert('Analysis Failed', 'Could not complete analysis');
      }
    } catch (error) {
      Alert.alert('Error', String(error));
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Scan ECG</Text>

      {isDemo ? (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Demo Mode</Text>
          <Text style={styles.description}>
            Generate a synthetic normal sinus rhythm ECG and run the full analysis pipeline.
          </Text>

          <TouchableOpacity
            style={[styles.demoButton, isAnalyzing && styles.demoButtonDisabled]}
            onPress={handleUseDemoECG}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <ActivityIndicator color="#071312" style={{ marginRight: 8 }} />
                <Text style={styles.demoButtonText}>Analyzing...</Text>
              </>
            ) : (
              <Text style={styles.demoButtonText}>🧪 Use Demo ECG</Text>
            )}
          </TouchableOpacity>

          <Text style={styles.note}>
            This synthetic ECG is intended for testing the analysis pipeline only.
            Real clinical use requires actual patient recordings.
          </Text>
        </View>
      ) : (
        <View style={styles.section}>
          <Text style={styles.placeholder}>Camera capture — implementation pending (Phase 5)</Text>
          <Text style={styles.note}>
            HIPAA 5.3.6: Captured images will be processed in memory only.
            No images saved to Gallery/Photos.
          </Text>

          <TouchableOpacity
            style={styles.demoLinkButton}
            onPress={() => router.push('/scan?demo=true')}
          >
            <Text style={styles.demoLinkText}>Try Demo Mode</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#071312',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#00E5B0',
    marginBottom: 16,
  },
  section: {
    flex: 1,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#00E5B0',
    marginBottom: 12,
  },
  description: {
    fontSize: 14,
    color: '#5A8A85',
    marginBottom: 20,
    lineHeight: 20,
  },
  placeholder: {
    fontSize: 14,
    color: '#5A8A85',
    marginBottom: 8,
  },
  note: {
    fontSize: 12,
    color: '#3D6662',
    fontStyle: 'italic',
    marginTop: 16,
    lineHeight: 16,
  },
  demoButton: {
    backgroundColor: '#00E5B0',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 20,
    alignItems: 'center',
    marginBottom: 20,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  demoButtonDisabled: {
    opacity: 0.6,
  },
  demoButtonText: {
    color: '#071312',
    fontSize: 16,
    fontWeight: '600',
  },
  demoLinkButton: {
    marginTop: 24,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: '#00E5B0',
    borderRadius: 8,
    alignItems: 'center',
  },
  demoLinkText: {
    color: '#00E5B0',
    fontSize: 14,
    fontWeight: '600',
  },
});

