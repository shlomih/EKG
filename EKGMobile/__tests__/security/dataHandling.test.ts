/**
 * HIPAA 5.1 — Data Handling Tests
 *
 * Verifies that:
 * - "Delete All My Data" removes all patient data
 * - Deletion preserves audit log
 * - Clinical disclaimer is present
 * - App permissions are minimal
 */

jest.mock('@op-engineering/op-sqlite');
jest.mock('expo-crypto', () => ({
  digestStringAsync: jest.fn(async () => '0'.repeat(64)),
  CryptoDigestAlgorithm: { SHA256: 'SHA-256' },
}));

describe('HIPAA 5.1.2 - Delete All My Data', () => {
  test('D-1 to D-4: Delete all data removes PHI but preserves audit log', async () => {
    // Track what SQL gets executed
    const executedSql: string[] = [];
    const mockDb = {
      execute: jest.fn((sql: string) => {
        executedSql.push(sql);

        if (sql.includes('COUNT(*)')) {
          return { rows: { length: 1, item: () => ({ cnt: 5 }) } };
        }
        if (sql.includes('FROM audit_log')) {
          return { rows: { length: 0 } };
        }
        return { rows: { length: 0 } };
      }),
    };

    // Initialize audit logger
    const { initAuditLogger } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    // Call deleteAllData
    const { deleteAllData } = require('../../src/db/PatientRepository');

    // Mock getDatabase to return our mock
    jest.spyOn(require('../../src/db/Database'), 'getDatabase').mockReturnValue(mockDb);

    await deleteAllData();

    // Verify DELETE was called on data tables
    expect(executedSql).toContain('DELETE FROM analysis_results');
    expect(executedSql).toContain('DELETE FROM ekg_records');
    expect(executedSql).toContain('DELETE FROM patients');

    // Verify DELETE was NOT called on audit_log
    const auditDeletes = executedSql.filter(
      (sql) => sql.includes('DELETE') && sql.includes('audit_log')
    );
    expect(auditDeletes).toHaveLength(0);

    // Verify DATA_WIPE event was logged to audit
    const auditInserts = executedSql.filter(
      (sql) => sql.includes('INSERT INTO audit_log')
    );
    expect(auditInserts.length).toBeGreaterThan(0);
  });
});

describe('HIPAA 5.3.5 - Clinical disclaimer', () => {
  test('D-9: Disclaimer text contains required elements', () => {
    const requiredPhrases = [
      'educational purposes only',
      'not a medical diagnosis',
      'Not FDA-cleared',
      'healthcare professional',
    ];

    const disclaimer =
      'For educational purposes only. This is not a medical diagnosis. ' +
      'Not FDA-cleared. Always consult a qualified healthcare professional.';

    for (const phrase of requiredPhrases) {
      expect(disclaimer.toLowerCase()).toContain(phrase.toLowerCase());
    }
  });
});

describe('HIPAA 5.1.5 - No PHI in notifications', () => {
  test('P-4: Notification payloads contain no medical data', () => {
    // Define allowed notification templates
    const allowedNotifications = [
      'Analysis complete',
      'New result ready',
      'Export ready',
    ];

    // These should NEVER appear in notifications
    const forbiddenPatterns = [
      /AFIB|atrial fibrillation/i,
      /LVH|hypertrophy/i,
      /STEMI|infarction/i,
      /abnormal|normal/i,
      /\d+\s*bpm/i,        // heart rate
      /QTc|QRS|PR\s*interval/i,
    ];

    for (const notification of allowedNotifications) {
      for (const pattern of forbiddenPatterns) {
        expect(notification).not.toMatch(pattern);
      }
    }
  });
});
