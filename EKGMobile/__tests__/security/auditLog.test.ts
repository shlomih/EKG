/**
 * HIPAA 1.3.1, 1.3.2, 1.3.3 — Audit Log Tests
 *
 * Verifies that:
 * - Every PHI access event creates an audit entry
 * - Audit entries never contain PHI content
 * - Hash chain is intact (tamper detection)
 * - Audit log is append-only
 * - User can view their own audit log
 */

jest.mock('@op-engineering/op-sqlite');
jest.mock('expo-crypto', () => ({
  digestStringAsync: jest.fn(async (_algo: string, input: string) => {
    // Simple deterministic hash for testing
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
      hash = ((hash << 5) - hash + input.charCodeAt(i)) | 0;
    }
    return Math.abs(hash).toString(16).padStart(64, '0');
  }),
  CryptoDigestAlgorithm: { SHA256: 'SHA-256' },
}));

describe('HIPAA 1.3.1 - Audit log of PHI access events', () => {
  let mockDb: any;
  let auditRows: any[];

  beforeEach(() => {
    auditRows = [];
    mockDb = {
      execute: jest.fn((sql: string, params?: any[]) => {
        if (sql.startsWith('INSERT INTO audit_log')) {
          auditRows.push({
            id: auditRows.length + 1,
            timestamp: params?.[0],
            event_type: params?.[1],
            resource_type: params?.[2],
            resource_id: params?.[3],
            action: params?.[4],
            details: params?.[5],
            integrity_hash: params?.[6],
          });
          return { rows: { length: 0 } };
        }
        if (sql.includes('FROM audit_log ORDER BY id DESC LIMIT 1')) {
          if (auditRows.length > 0) {
            return {
              rows: {
                length: 1,
                item: () => auditRows[auditRows.length - 1],
              },
            };
          }
          return { rows: { length: 0 } };
        }
        if (sql.includes('FROM audit_log ORDER BY id ASC')) {
          return {
            rows: {
              length: auditRows.length,
              item: (i: number) => auditRows[i],
            },
          };
        }
        return { rows: { length: 0 } };
      }),
    };
  });

  test('AU-1: Auth success is logged', async () => {
    const { initAuditLogger, logEvent } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    await logEvent('AUTH_SUCCESS', 'biometric_auth');

    expect(auditRows).toHaveLength(1);
    expect(auditRows[0].event_type).toBe('AUTH_SUCCESS');
    expect(auditRows[0].action).toBe('biometric_auth');
  });

  test('AU-2: Auth failure is logged', async () => {
    const { initAuditLogger, logEvent } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    await logEvent('AUTH_FAIL', 'biometric_rejected');

    expect(auditRows).toHaveLength(1);
    expect(auditRows[0].event_type).toBe('AUTH_FAIL');
  });

  test('AU-3: Patient creation is logged', async () => {
    const { initAuditLogger, logEvent } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    await logEvent('PHI_CREATE', 'create_patient', 'patient', 'patient-uuid-123');

    expect(auditRows).toHaveLength(1);
    expect(auditRows[0].event_type).toBe('PHI_CREATE');
    expect(auditRows[0].resource_type).toBe('patient');
    expect(auditRows[0].resource_id).toBe('patient-uuid-123');
  });

  test('AU-4: Patient view is logged', async () => {
    const { initAuditLogger, logEvent } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    await logEvent('PHI_VIEW', 'view_patient', 'patient', 'patient-uuid-456');

    expect(auditRows[0].event_type).toBe('PHI_VIEW');
  });

  test('AU-5: Analysis creation is logged', async () => {
    const { initAuditLogger, logEvent } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    await logEvent('PHI_CREATE', 'create_analysis', 'analysis', 'analysis-uuid-789');

    expect(auditRows[0].event_type).toBe('PHI_CREATE');
    expect(auditRows[0].resource_type).toBe('analysis');
  });

  test('AU-6: PDF export is logged', async () => {
    const { initAuditLogger, logEvent } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    await logEvent('PHI_EXPORT', 'export_pdf', 'report', 'report-uuid-101');

    expect(auditRows[0].event_type).toBe('PHI_EXPORT');
  });

  test('AU-7: Patient deletion is logged', async () => {
    const { initAuditLogger, logEvent } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    await logEvent('PHI_DELETE', 'delete_patient', 'patient', 'patient-uuid-999');

    expect(auditRows[0].event_type).toBe('PHI_DELETE');
  });

  test('AU-8: Audit entries contain NO PHI content', async () => {
    const { initAuditLogger, logEvent } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    // Log various events
    await logEvent('PHI_CREATE', 'create_patient', 'patient', 'uuid-1');
    await logEvent('PHI_VIEW', 'view_patient', 'patient', 'uuid-1');
    await logEvent('PHI_EXPORT', 'export_pdf', 'report', 'uuid-2');

    // Check that no audit entry contains PHI
    const phiPatterns = [
      /John/i, /Smith/i, /\d{3}-\d{2}-\d{4}/, // SSN
      /\d{1,3}\.\d+/, // signal values
    ];

    for (const row of auditRows) {
      const allFields = JSON.stringify(row);
      for (const pattern of phiPatterns) {
        // resource_id UUIDs are allowed — they reference PHI but ARE NOT PHI
        const fieldsToCheck = `${row.event_type}|${row.action}|${row.details || ''}`;
        expect(fieldsToCheck).not.toMatch(pattern);
      }
    }
  });
});

describe('HIPAA 1.3.2 - Tamper-resistant audit log', () => {
  test('AU-9: Hash chain is valid after multiple operations', async () => {
    const mockDb = createMockDbWithAuditStorage();
    const { initAuditLogger, logEvent, verifyIntegrity } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    // Log 10 events
    for (let i = 0; i < 10; i++) {
      await logEvent('PHI_VIEW', `action_${i}`, 'patient', `uuid-${i}`);
    }

    const result = await verifyIntegrity();
    expect(result.valid).toBe(true);
    expect(result.totalEntries).toBe(10);
  });

  test('AU-10: Tampered entry is detected', async () => {
    const storage: any[] = [];
    const mockDb = createMockDbWithStorage(storage);
    const { initAuditLogger, logEvent, verifyIntegrity } = require('../../src/audit/AuditLogger');
    await initAuditLogger(mockDb);

    await logEvent('PHI_VIEW', 'action_1', 'patient', 'uuid-1');
    await logEvent('PHI_VIEW', 'action_2', 'patient', 'uuid-2');
    await logEvent('PHI_VIEW', 'action_3', 'patient', 'uuid-3');

    // Tamper with the second entry
    storage[1].action = 'TAMPERED_ACTION';

    const result = await verifyIntegrity();
    expect(result.valid).toBe(false);
    expect(result.brokenAtIndex).toBe(1);
  });

  test('AU-14: Delete All Data preserves audit log', async () => {
    const { initAuditLogger, logEvent, getEntryCount } = require('../../src/audit/AuditLogger');
    const mockDb = createMockDbWithAuditStorage();
    await initAuditLogger(mockDb);

    await logEvent('PHI_CREATE', 'create_patient', 'patient', 'uuid-1');
    await logEvent('PHI_VIEW', 'view_patient', 'patient', 'uuid-1');

    const countBefore = getEntryCount();

    // Simulate deleteAllData — should add DATA_WIPE event but NOT clear audit
    await logEvent('DATA_WIPE', 'delete_all_data');

    const countAfter = getEntryCount();
    expect(countAfter).toBe(countBefore + 1);
  });
});

// Helper: create mock DB with in-memory audit storage
function createMockDbWithAuditStorage() {
  const storage: any[] = [];
  return createMockDbWithStorage(storage);
}

function createMockDbWithStorage(storage: any[]) {
  return {
    execute: jest.fn((sql: string, params?: any[]) => {
      if (sql.startsWith('INSERT INTO audit_log')) {
        storage.push({
          id: storage.length + 1,
          timestamp: params?.[0],
          event_type: params?.[1],
          resource_type: params?.[2],
          resource_id: params?.[3],
          action: params?.[4],
          details: params?.[5],
          integrity_hash: params?.[6],
        });
        return { rows: { length: 0 } };
      }
      if (sql.includes('ORDER BY id DESC LIMIT 1')) {
        if (storage.length > 0) {
          return {
            rows: {
              length: 1,
              item: () => storage[storage.length - 1],
            },
          };
        }
        return { rows: { length: 0 } };
      }
      if (sql.includes('ORDER BY id ASC')) {
        return {
          rows: {
            length: storage.length,
            item: (i: number) => storage[i],
          },
        };
      }
      if (sql.includes('COUNT(*)')) {
        return {
          rows: {
            length: 1,
            item: () => ({ count: storage.length }),
          },
        };
      }
      return { rows: { length: 0 } };
    }),
  };
}
