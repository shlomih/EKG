/**
 * AnalysisRepository — CRUD operations for analysis results.
 *
 * All operations are logged to the HIPAA audit trail.
 * Deletion cascades from ekg_records deletion.
 */

import { v4 as uuidv4 } from 'uuid';
import { getDatabase } from './Database';
import { logEvent } from '../audit/AuditLogger';
import type {
  AnalysisResult,
  IntervalMeasurements,
  ClinicalRuleResult,
  STTerritoryResult,
} from '../types/ECGRecord';

/**
 * Create a new analysis result.
 */
export async function saveAnalysis(
  analysis: Omit<AnalysisResult, 'analysis_id' | 'created_at'>,
): Promise<string> {
  const db = getDatabase();
  const analysisId = uuidv4();

  db.execute(
    `INSERT INTO analysis_results
     (analysis_id, ekg_id, model_version, model_hash, primary_condition,
      conditions_json, scores_json, intervals_json, clinical_rules_json,
      st_territory_json, disclaimer_shown)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      analysisId,
      analysis.ekg_id,
      analysis.model_version,
      analysis.model_hash,
      analysis.primary_condition,
      JSON.stringify(analysis.conditions),
      JSON.stringify(analysis.scores),
      analysis.intervals ? JSON.stringify(analysis.intervals) : null,
      analysis.clinical_rules ? JSON.stringify(analysis.clinical_rules) : null,
      analysis.st_territory ? JSON.stringify(analysis.st_territory) : null,
      analysis.disclaimer_shown ? 1 : 0,
    ],
  );

  await logEvent('PHI_CREATE', 'create_analysis', 'analysis', analysisId);
  return analysisId;
}

/**
 * Get a single analysis result by ID.
 */
export async function getAnalysis(analysisId: string): Promise<AnalysisResult | null> {
  const db = getDatabase();
  const result = db.execute('SELECT * FROM analysis_results WHERE analysis_id = ?', [
    analysisId,
  ]);

  if (!result.rows || result.rows.length === 0) return null;

  await logEvent('PHI_VIEW', 'view_analysis', 'analysis', analysisId);
  return rowToAnalysis(result.rows.item(0));
}

/**
 * List all analyses for an ECG record.
 */
export async function listAnalysesByRecord(ekgId: string): Promise<AnalysisResult[]> {
  const db = getDatabase();
  const result = db.execute(
    'SELECT * FROM analysis_results WHERE ekg_id = ? ORDER BY created_at DESC',
    [ekgId],
  );

  if (!result.rows) return [];

  const count = result.rows.length;
  await logEvent(
    'PHI_VIEW',
    'list_analyses',
    'analysis',
    null,
    `ekg:${ekgId},count:${count}`,
  );

  const analyses: AnalysisResult[] = [];
  for (let i = 0; i < result.rows.length; i++) {
    analyses.push(rowToAnalysis(result.rows.item(i)));
  }
  return analyses;
}

/**
 * Get the most recent analysis for an ECG record.
 */
export async function getLatestAnalysis(ekgId: string): Promise<AnalysisResult | null> {
  const db = getDatabase();
  const result = db.execute(
    'SELECT * FROM analysis_results WHERE ekg_id = ? ORDER BY created_at DESC LIMIT 1',
    [ekgId],
  );

  if (!result.rows || result.rows.length === 0) return null;

  const analysis = rowToAnalysis(result.rows.item(0));
  await logEvent('PHI_VIEW', 'view_latest_analysis', 'analysis', analysis.analysis_id);
  return analysis;
}

function rowToAnalysis(row: any): AnalysisResult {
  return {
    analysis_id: row.analysis_id,
    ekg_id: row.ekg_id,
    model_version: row.model_version,
    model_hash: row.model_hash,
    primary_condition: row.primary_condition,
    conditions: JSON.parse(row.conditions_json),
    scores: JSON.parse(row.scores_json),
    intervals: row.intervals_json ? JSON.parse(row.intervals_json) : null,
    clinical_rules: row.clinical_rules_json ? JSON.parse(row.clinical_rules_json) : null,
    st_territory: row.st_territory_json ? JSON.parse(row.st_territory_json) : null,
    disclaimer_shown: !!row.disclaimer_shown,
    created_at: row.created_at,
  };
}
