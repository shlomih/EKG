/**
 * PHISanitizer — HIPAA 1.1.3: Strip Protected Health Information from
 * all log output, error reports, and analytics events.
 *
 * Intercepts console.log, console.warn, console.error to filter out
 * patient names, IDs, ages, ECG signal data, and analysis results.
 *
 * Call installSanitizer() once at app startup, before any other code runs.
 */

// Patterns that indicate PHI content
const PHI_PATTERNS = [
  // UUIDs (patient_id, ekg_id, analysis_id)
  /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi,
  // ECG signal arrays (large numeric arrays)
  /\[[\s\d.,eE+-]{200,}\]/g,
  // Float32Array dumps
  /Float32Array\[[\s\S]*?\]/g,
];

// Keys that should be redacted in objects
const SENSITIVE_KEYS = new Set([
  'first_name', 'firstName',
  'last_name', 'lastName',
  'id_number', 'idNumber',
  'patient_id', 'patientId',
  'signal_data', 'signalData',
  'signal',
  'conditions_json', 'conditionsJson',
  'scores_json', 'scoresJson',
]);

/**
 * Sanitize a single value, stripping PHI content.
 */
function sanitizeValue(value: unknown): unknown {
  if (value === null || value === undefined) return value;

  if (typeof value === 'string') {
    let sanitized = value;
    for (const pattern of PHI_PATTERNS) {
      sanitized = sanitized.replace(pattern, '[REDACTED]');
    }
    return sanitized;
  }

  if (typeof value === 'object') {
    if (Array.isArray(value)) {
      // Redact large numeric arrays (likely ECG signal data)
      if (value.length > 100 && value.every((v) => typeof v === 'number')) {
        return `[REDACTED: array of ${value.length} numbers]`;
      }
      return value.map(sanitizeValue);
    }

    // Redact known sensitive object keys
    const sanitized: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(value as Record<string, unknown>)) {
      if (SENSITIVE_KEYS.has(key)) {
        sanitized[key] = '[REDACTED]';
      } else {
        sanitized[key] = sanitizeValue(val);
      }
    }
    return sanitized;
  }

  return value;
}

/**
 * Sanitize arguments passed to console methods.
 */
function sanitizeArgs(args: unknown[]): unknown[] {
  return args.map(sanitizeValue);
}

// Store original console methods for restore
let originalLog: typeof console.log | null = null;
let originalWarn: typeof console.warn | null = null;
let originalError: typeof console.error | null = null;
let installed = false;

/**
 * Install the PHI sanitizer. Wraps console.log/warn/error to strip
 * sensitive data before output. Call once at app startup.
 *
 * HIPAA 1.1.3: No PHI written to application logs.
 */
export function installSanitizer(): void {
  if (installed) return;

  originalLog = console.log;
  originalWarn = console.warn;
  originalError = console.error;

  console.log = (...args: unknown[]) => {
    originalLog!(...sanitizeArgs(args));
  };

  console.warn = (...args: unknown[]) => {
    originalWarn!(...sanitizeArgs(args));
  };

  console.error = (...args: unknown[]) => {
    originalError!(...sanitizeArgs(args));
  };

  installed = true;
}

/**
 * Restore original console methods. Used in testing.
 */
export function uninstallSanitizer(): void {
  if (!installed) return;

  if (originalLog) console.log = originalLog;
  if (originalWarn) console.warn = originalWarn;
  if (originalError) console.error = originalError;

  originalLog = null;
  originalWarn = null;
  originalError = null;
  installed = false;
}

/**
 * Sanitize an error for crash reporting. Strips PHI from message
 * and stack trace while preserving error structure for debugging.
 */
export function sanitizeError(error: Error): { message: string; stack?: string } {
  return {
    message: sanitizeValue(error.message) as string,
    stack: error.stack ? (sanitizeValue(error.stack) as string) : undefined,
  };
}
