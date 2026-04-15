/**
 * Analysis result types for V3 multilabel model
 * Maps 26-class predictions to clinical display data
 */

import {
  V3_CODES,
  V3_URGENCY,
  V3_CONDITION_DESCRIPTIONS,
  V3_CLINICAL_GUIDANCE,
  getUrgencyColor,
  getUrgencyLabel,
  type ConditionCode,
} from '@/src/ml/ConditionMetadata';

/**
 * A detected condition with clinical metadata
 */
export interface DetectedCondition {
  code: string;
  name: string;
  probability: number;
  urgency: 0 | 1 | 2 | 3;
  urgencyLabel: string;
  urgencyColor: string;
  action: string;
  note: string;
}

/**
 * Display-ready analysis results
 * Used by result screens and sharing
 */
export interface AnalysisDisplay {
  conditions: DetectedCondition[];
  modelVersion: string;
  timestamp: string;
  disclaimerShown: boolean;
}

/**
 * Convert a condition code + probability to a DetectedCondition
 * using clinical metadata lookups
 */
export function toDetectedCondition(
  code: string,
  probability: number,
): DetectedCondition {
  const isKnown = (V3_CODES as readonly string[]).includes(code);

  if (!isKnown) {
    return {
      code,
      name: `Unknown (${code})`,
      probability,
      urgency: 0,
      urgencyLabel: 'Normal',
      urgencyColor: '#00E5B0',
      action: 'Monitor regularly',
      note: '',
    };
  }

  const cc = code as ConditionCode;
  const urgency = (V3_URGENCY[cc] ?? 0) as 0 | 1 | 2 | 3;
  const guidance = V3_CLINICAL_GUIDANCE[cc];

  return {
    code,
    name: V3_CONDITION_DESCRIPTIONS[cc],
    probability,
    urgency,
    urgencyLabel: getUrgencyLabel(urgency),
    urgencyColor: getUrgencyColor(urgency),
    action: guidance.action,
    note: guidance.note,
  };
}
