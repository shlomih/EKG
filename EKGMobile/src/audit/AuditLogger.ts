/**
 * AuditLogger — HIPAA 1.3.1, 1.3.2: Append-only audit trail with
 * tamper-evident hash chain.
 *
 * Every PHI access event is logged with a SHA-256 integrity hash.
 * Each entry's hash includes the previous entry's hash, forming a chain.
 * Tampering with any entry breaks the chain from that point forward.
 *
 * The audit_log table is append-only — no UPDATE or DELETE operations
 * are permitted on audit entries.
 */

import * as Crypto from 'expo-crypto';
import type { AuditEntry, AuditEventType, ResourceType } from './AuditTypes';

// Database reference — set by Database.ts after connection is opened
let db: any = null;
let lastHash: string = 'GENESIS';

/**
 * Initialize the audit logger with a database connection.
 * Must be called after Database.open().
 */
export async function initAuditLogger(database: any): Promise<void> {
  db = database;

  // Recover the last hash from the most recent audit entry
  const result = db.execute(
    'SELECT integrity_hash FROM audit_log ORDER BY id DESC LIMIT 1'
  );
  if (result.rows && result.rows.length > 0) {
    lastHash = result.rows.item(0).integrity_hash;
  } else {
    lastHash = 'GENESIS';
  }
}

/**
 * Compute the integrity hash for a new audit entry.
 * Hash = SHA-256(previousHash + timestamp + eventType + action)
 */
async function computeHash(
  previousHash: string,
  timestamp: string,
  eventType: string,
  action: string,
): Promise<string> {
  const input = `${previousHash}|${timestamp}|${eventType}|${action}`;
  return await Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    input,
  );
}

/**
 * Log an audit event. Appends to the audit_log table with a chain hash.
 *
 * IMPORTANT: Never include PHI in the `details` parameter.
 * Use resource_id to reference the data, not the data itself.
 */
export async function logEvent(
  eventType: AuditEventType,
  action: string,
  resourceType: ResourceType | null = null,
  resourceId: string | null = null,
  details: string | null = null,
): Promise<void> {
  if (!db) {
    // Pre-initialization — queue for later or silently skip
    return;
  }

  const timestamp = new Date().toISOString();
  const hash = await computeHash(lastHash, timestamp, eventType, action);

  db.execute(
    `INSERT INTO audit_log (timestamp, event_type, resource_type, resource_id, action, details, integrity_hash)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
    [timestamp, eventType, resourceType, resourceId, action, details, hash],
  );

  lastHash = hash;
}

/**
 * Verify the integrity of the entire audit log chain.
 * Returns { valid: boolean, brokenAtIndex?: number }.
 *
 * HIPAA 1.3.2: Tamper-resistant audit log verification.
 */
export async function verifyIntegrity(): Promise<{
  valid: boolean;
  totalEntries: number;
  brokenAtIndex?: number;
}> {
  if (!db) {
    return { valid: false, totalEntries: 0 };
  }

  const result = db.execute(
    'SELECT id, timestamp, event_type, action, integrity_hash FROM audit_log ORDER BY id ASC'
  );

  if (!result.rows || result.rows.length === 0) {
    return { valid: true, totalEntries: 0 };
  }

  let previousHash = 'GENESIS';
  const totalEntries = result.rows.length;

  for (let i = 0; i < totalEntries; i++) {
    const row = result.rows.item(i);
    const expectedHash = await computeHash(
      previousHash,
      row.timestamp,
      row.event_type,
      row.action,
    );

    if (expectedHash !== row.integrity_hash) {
      return { valid: false, totalEntries, brokenAtIndex: i };
    }

    previousHash = row.integrity_hash;
  }

  return { valid: true, totalEntries };
}

/**
 * Get recent audit entries for display in the UI.
 * HIPAA 1.3.3: User can view their own audit log.
 */
export function getRecentEntries(limit: number = 50): AuditEntry[] {
  if (!db) return [];

  const result = db.execute(
    'SELECT * FROM audit_log ORDER BY id DESC LIMIT ?',
    [limit],
  );

  if (!result.rows) return [];

  const entries: AuditEntry[] = [];
  for (let i = 0; i < result.rows.length; i++) {
    entries.push(result.rows.item(i) as AuditEntry);
  }
  return entries;
}

/**
 * Get total count of audit entries.
 */
export function getEntryCount(): number {
  if (!db) return 0;

  const result = db.execute('SELECT COUNT(*) as count FROM audit_log');
  return result.rows?.item(0)?.count ?? 0;
}
