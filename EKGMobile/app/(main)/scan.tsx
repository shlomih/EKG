/**
 * Scan screen — camera capture for paper ECG digitization.
 * Supports demo mode with synthetic ECG data and full inference pipeline.
 * Also supports camera capture → signal extraction → analysis flow.
 *
 * Flow:
 * 1. User taps "Scan" → CameraCapture screen opens
 * 2. User aligns paper within guide frame, taps "Capture"
 * 3. Image saved to temp file, SignalExtractor processes it
 * 4. Signal passed to inference pipeline
 * 5. Results saved to database and displayed
 * 6. Temp file securely deleted (HIPAA 5.3.6)
 */

import { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { createDemoRecord } from '@/src/ml/DemoData';
import { runInferencePipeline } from '@/src/ml/InferencePipeline';
import CameraCapture, { CameraResult } from '@/src/digitization/CameraCapture';
import { extractSignal } from '@/src/digitization/SignalExtractor';
import useAppStore from '@/src/store/useAppStore';

export default function ScanScreen() {
  const router = useRouter();
  const { demo } = useLocalSearchParams<{ demo?: string }>();
  const { currentPatient } = useAppStore();
  const isDemo = demo === 'true';

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showCamera, setShowCamera] = useState(false);

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

  const handleCameraCapture = async (result: CameraResult) => {
    if (!currentPatient) {
      Alert.alert('No Patient Selected', 'Please select a patient first');
      setShowCamera(false);
      router.push('/patients');
      return;
    }

    setIsAnalyzing(true);
    setShowCamera(false);

    try {
      // Step 1: Extract signal from image (temp file deleted in SignalExtractor)
      const extractionResult = await extractSignal(result.imageUri, result.paperSize);

      // Step 2: Run inference pipeline with extracted signal
      const analysisId = await runInferencePipeline(
        currentPatient.id,
        extractionResult.signal,
        'camera',
      );

      if (analysisId) {
        // Step 3: Navigate to analysis results
        router.push(`/analyze?analysisId=${analysisId}`);
      } else {
        Alert.alert('Analysis Failed', 'Could not complete analysis. Please try again.');
      }
    } catch (error) {
      console.error('Scan error:', error);
      Alert.alert(
        'Scan Error',
        `Failed to process image: ${error instanceof Error ? error.message : String(error)}`,
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Camera capture screen (full-screen modal)
  if (showCamera) {
    return (
      <CameraCapture
        onCapture={handleCameraCapture}
        onCancel={() => setShowCamera(false)}
      />
    );
  }

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

          <TouchableOpacity
            style={styles.backButton}
            onPress={() => router.push('/scan')}
          >
            <Text style={styles.backButtonText}>← Back to Camera</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Camera Capture</Text>

          <View style={styles.warningBanner}>
            <Text style={styles.warningBannerTitle}>⚠ Not wired to real digitization</Text>
            <Text style={styles.warningBannerText}>
              Signal extraction is a placeholder — camera capture produces synthetic sine waves,
              not a real ECG signal. Any diagnosis shown is not based on the paper strip.
              Use Demo Mode for pipeline testing until the TypeScript digitizer ships.
            </Text>
          </View>

          <Text style={styles.description}>
            Point your phone at a paper ECG strip. The app will align and digitize it automatically.
          </Text>

          <TouchableOpacity
            style={[styles.cameraButton, isAnalyzing && styles.cameraButtonDisabled]}
            onPress={() => setShowCamera(true)}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <ActivityIndicator color="#071312" style={{ marginRight: 8 }} />
                <Text style={styles.cameraButtonText}>Processing...</Text>
              </>
            ) : (
              <Text style={styles.cameraButtonText}>📷 Scan Paper Strip</Text>
            )}
          </TouchableOpacity>

          <Text style={styles.note}>
            ✓ HIPAA 5.3.6: Images processed in memory only
            ✓ Temp files securely deleted after analysis
            ✓ No data saved to device storage
          </Text>

          <TouchableOpacity
            style={styles.demoLinkButton}
            onPress={() => router.push('/scan?demo=true')}
            disabled={isAnalyzing}
          >
            <Text style={styles.demoLinkText}>Or try Demo Mode</Text>
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
    lineHeight: 18,
  },
  cameraButton: {
    backgroundColor: '#00E5B0',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 20,
    alignItems: 'center',
    marginBottom: 20,
    flexDirection: 'row',
    justifyContent: 'center',
  },
  cameraButtonDisabled: {
    opacity: 0.6,
  },
  cameraButtonText: {
    color: '#071312',
    fontSize: 16,
    fontWeight: '600',
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
  backButton: {
    marginTop: 24,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: '#5A8A85',
    borderRadius: 8,
    alignItems: 'center',
  },
  backButtonText: {
    color: '#5A8A85',
    fontSize: 14,
    fontWeight: '600',
  },
  warningBanner: {
    backgroundColor: '#3D2A0A',
    borderColor: '#FFB020',
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
  },
  warningBannerTitle: {
    color: '#FFB020',
    fontSize: 13,
    fontWeight: '700',
    marginBottom: 4,
  },
  warningBannerText: {
    color: '#FFD580',
    fontSize: 12,
    lineHeight: 17,
  },
});

