/**
 * Database — HIPAA 1.1.1: AES-256 encrypted SQLite via SQLCipher.
 *
 * The database file is encrypted at rest using a key stored in the
 * device's hardware security module (Android Keystore / iOS Secure Enclave).
 *
 * The database is opened ONLY after successful biometric authentication
 * (via AuthProvider), which retrieves the key from the secure keystore.
 */

import { open, type DB } from '@op-engineering/op-sqlite';
import { CREATE_TABLES, CREATE_INDEXES, DEFAULT_CONFIG, SCHEMA_VERSION } from './schema';
import { initAuditLogger } from '../audit/AuditLogger';

let db: DB | null = null;

/**
 * Open the encrypted database with the given key.
 * Called by AuthProvider after successful biometric authentication.
 *
 * @param encryptionKey - AES-256 key from KeyManager (hex string)
 */
export async function openDatabase(encryptionKey: string): Promise<DB> {
  if (db) return db;

  db = open({
    name: 'ekg_platform.db',
    encryptionKey,
  });

  // Enable foreign keys for CASCADE deletes (HIPAA 5.1.2)
  db.execute('PRAGMA foreign_keys = ON');

  // Create tables if they don't exist
  const statements = CREATE_TABLES.split(';')
    .map(s => s.trim())
    .filter(s => s.length > 0);

  for (const stmt of statements) {
    db.execute(stmt);
  }

  // Create indexes
  const indexes = CREATE_INDEXES.split(';')
    .map(s => s.trim())
    .filter(s => s.length > 0);

  for (const idx of indexes) {
    db.execute(idx);
  }

  // Initialize default config if empty
  const configCount = db.execute('SELECT COUNT(*) as cnt FROM app_config');
  if (configCount.rows?.item(0)?.cnt === 0) {
    for (const [key, value] of Object.entries(DEFAULT_CONFIG)) {
      db.execute(
        'INSERT OR IGNORE INTO app_config (key, value) VALUES (?, ?)',
        [key, value],
      );
    }
  }

  // Run migrations if needed
  await runMigrations(db);

  // Initialize audit logger with DB connection
  await initAuditLogger(db);

  return db;
}

/**
 * Get the current database connection.
 * Throws if the database has not been opened (user not authenticated).
 */
export function getDatabase(): DB {
  if (!db) {
    throw new Error(
      'Database not open. User must authenticate first. ' +
      'This error indicates a PHI access attempt without authentication (HIPAA violation).'
    );
  }
  return db;
}

/**
 * Close the database connection. Called when the session is locked.
 * After closing, the key is no longer in memory — re-auth is required.
 */
export function closeDatabase(): void {
  if (db) {
    db.close();
    db = null;
  }
}

/**
 * Check if the database is currently open.
 */
export function isDatabaseOpen(): boolean {
  return db !== null;
}

/**
 * Run schema migrations. Handles version upgrades gracefully.
 */
async function runMigrations(database: DB): Promise<void> {
  // Check current schema version
  const versionResult = database.execute(
    'SELECT version FROM schema_version ORDER BY version DESC LIMIT 1'
  );

  let currentVersion = 0;
  if (versionResult.rows && versionResult.rows.length > 0) {
    currentVersion = versionResult.rows.item(0).version;
  }

  if (currentVersion >= SCHEMA_VERSION) return;

  // Apply migrations sequentially
  // Migration 0 → 1: Initial schema (already created above)
  if (currentVersion < 1) {
    database.execute(
      'INSERT OR REPLACE INTO schema_version (version) VALUES (?)',
      [1],
    );
  }

  // Future migrations go here:
  // if (currentVersion < 2) { ... migrate to v2 ... }
}

/**
 * Get a config value from app_config.
 */
export function getConfig(key: string): string | null {
  const d = getDatabase();
  const result = d.execute('SELECT value FROM app_config WHERE key = ?', [key]);
  return result.rows?.item(0)?.value ?? null;
}

/**
 * Set a config value in app_config.
 */
export function setConfig(key: string, value: string): void {
  const d = getDatabase();
  d.execute(
    'INSERT OR REPLACE INTO app_config (key, value, updated_at) VALUES (?, ?, datetime("now"))',
    [key, value],
  );
}
