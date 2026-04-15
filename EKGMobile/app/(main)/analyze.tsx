/**
 * Analysis results screen — displays ECG classification results.
 * Clinical disclaimer shown on every result (HIPAA 5.3.5 / FDA 7.1).
 *
 * Route params: { analysisId: string }
 */

import { useEffect, useState } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, ActivityIndicator, SafeAreaView } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { getAnalysis } from '../../src/db/AnalysisRepository';
import { type AnalysisResult, type IntervalMeasurements } from '../../src/types/ECGRecord';
import { toDetectedCondition, type DetectedCondition } from '../../src/types/Analysis';
import ConditionCard from '../../src/components/ConditionCard';
import DisclaimerBanner from '../../src/components/DisclaimerBanner';

export default function AnalyzeScreen() {
  const { analysisId } = useLocalSearchParams<{ analysisId: string }>();
  const router = useRouter();
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [conditions, setConditions] = useState<DetectedCondition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!analysisId) {
      setError('No analysis ID provided.');
      setLoading(false);
      return;
    }

    try {
      const result = getAnalysis(analysisId);
      if (!result) {
        setError('Analysis not found.');
        setLoading(false);
        return;
      }

      setAnalysis(result);

      // Convert scores to DetectedCondition[], sorted by urgency desc then probability desc
      const detected = Object.entries(result.scores)
        .map(([code, prob]) => toDetectedCondition(code, prob))
        .sort((a, b) => b.urgency - a.urgency || b.probability - a.probability);

      setConditions(detected);
    } catch {
      setError('Failed to load analysis results.');
    } finally {
      setLoading(false);
    }
  }, [analysisId]);

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color="#00E5B0" />
      </View>
    );
  }

  if (error || !analysis) {
    return (
      <SafeAreaView style={styles.container}>
        <TouchableOpacity onPress={() => router.back()}>
          <Text style={styles.backButton}>Back</Text>
        </TouchableOpacity>
        <View style={styles.centered}>
          <Text style={styles.errorText}>{error ?? 'Unknown error'}</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Text style={styles.backButton}>Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Analysis Results</Text>
        <Text style={styles.subtitle}>
          {analysis.model_version} &middot; {analysis.conditions.length} condition{analysis.conditions.length !== 1 ? 's' : ''} detected
        </Text>
      </View>

      {/* Interval Measurements */}
      {analysis.intervals && (
        <View style={styles.intervalsCard}>
          <Text style={styles.sectionTitle}>Intervals</Text>
          <View style={styles.intervalsGrid}>
            <IntervalItem label="HR" value={`${analysis.intervals.heart_rate}`} unit="bpm" />
            {analysis.intervals.pr_interval_ms != null && (
              <IntervalItem label="PR" value={`${analysis.intervals.pr_interval_ms}`} unit="ms" />
            )}
            {analysis.intervals.qrs_duration_ms != null && (
              <IntervalItem label="QRS" value={`${analysis.intervals.qrs_duration_ms}`} unit="ms" />
            )}
            {analysis.intervals.qtc_ms != null && (
              <IntervalItem label="QTc" value={`${analysis.intervals.qtc_ms}`} unit="ms" />
            )}
          </View>
        </View>
      )}

      {/* ST Territory */}
      {analysis.st_territory?.territory && (
        <View style={styles.territoryCard}>
          <Text style={styles.sectionTitle}>ST Territory</Text>
          <Text style={styles.territoryText}>
            {analysis.st_territory.territory} territory
          </Text>
          {analysis.st_territory.elevation_leads.length > 0 && (
            <Text style={styles.territoryDetail}>
              Elevation: {analysis.st_territory.elevation_leads.join(', ')}
            </Text>
          )}
          {analysis.st_territory.depression_leads.length > 0 && (
            <Text style={styles.territoryDetail}>
              Depression: {analysis.st_territory.depression_leads.join(', ')}
            </Text>
          )}
          {analysis.st_territory.reciprocal_changes && (
            <Text style={styles.reciprocalBadge}>Reciprocal changes present</Text>
          )}
        </View>
      )}

      {/* Conditions List */}
      <Text style={styles.sectionTitle}>Detected Conditions</Text>
      <FlatList
        data={conditions}
        keyExtractor={(item) => item.code}
        renderItem={({ item }) => <ConditionCard condition={item} />}
        contentContainerStyle={styles.listContent}
        ListEmptyComponent={
          <Text style={styles.emptyText}>No conditions detected above threshold.</Text>
        }
      />

      {/* Export button (stub) */}
      <TouchableOpacity style={styles.exportButton} onPress={() => {}}>
        <Text style={styles.exportButtonText}>Export PDF</Text>
      </TouchableOpacity>

      {/* Clinical Disclaimer — ALWAYS visible (HIPAA 5.3.5) */}
      <DisclaimerBanner />
    </SafeAreaView>
  );
}

function IntervalItem({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <View style={styles.intervalItem}>
      <Text style={styles.intervalLabel}>{label}</Text>
      <Text style={styles.intervalValue}>{value}</Text>
      <Text style={styles.intervalUnit}>{unit}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#071312', padding: 16 },
  centered: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  header: { marginBottom: 16 },
  backButton: { fontSize: 16, color: '#00E5B0', marginBottom: 12 },
  title: { fontSize: 24, fontWeight: '700', color: '#00E5B0', marginBottom: 4 },
  subtitle: { fontSize: 13, color: '#5A8A85' },
  errorText: { fontSize: 14, color: '#FF6B6B', textAlign: 'center' },
  sectionTitle: { fontSize: 14, fontWeight: '600', color: '#5A8A85', textTransform: 'uppercase', marginBottom: 8, marginTop: 12 },
  intervalsCard: {
    backgroundColor: '#0D1F1E', borderRadius: 8, padding: 14, marginBottom: 8,
    borderWidth: 1, borderColor: '#1E3533',
  },
  intervalsGrid: { flexDirection: 'row', justifyContent: 'space-around' },
  intervalItem: { alignItems: 'center' },
  intervalLabel: { fontSize: 12, color: '#5A8A85', marginBottom: 2 },
  intervalValue: { fontSize: 20, fontWeight: '700', color: '#E0E0E0' },
  intervalUnit: { fontSize: 11, color: '#3D6662' },
  territoryCard: {
    backgroundColor: '#0D1F1E', borderRadius: 8, padding: 14, marginBottom: 8,
    borderWidth: 1, borderColor: '#1E3533',
  },
  territoryText: { fontSize: 15, fontWeight: '600', color: '#E0E0E0', marginBottom: 4 },
  territoryDetail: { fontSize: 13, color: '#5A8A85', marginBottom: 2 },
  reciprocalBadge: { fontSize: 12, color: '#FF8800', fontWeight: '600', marginTop: 4 },
  listContent: { paddingBottom: 16 },
  emptyText: { fontSize: 14, color: '#5A8A85', textAlign: 'center', marginTop: 24 },
  exportButton: {
    backgroundColor: '#0D1F1E', borderRadius: 8, padding: 14, alignItems: 'center',
    borderWidth: 1, borderColor: '#00E5B0', marginBottom: 8,
  },
  exportButtonText: { fontSize: 15, fontWeight: '600', color: '#00E5B0' },
});
