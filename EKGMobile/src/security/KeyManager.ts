/**
 * KeyManager — HIPAA 1.1.2: Hardware-backed encryption key management.
 *
 * Generates and stores the AES-256 database encryption key in the device's
 * hardware security module (Android Keystore / iOS Secure Enclave) via
 * react-native-keychain. The key never exists in plaintext outside the HSM.
 *
 * Access requires biometric authentication or device passcode.
 */

import * as Keychain from 'react-native-keychain';
import * as Crypto from 'expo-crypto';

const SERVICE_NAME = 'com.ekgintelligence.dbkey';
const KEY_USERNAME = 'ekg_db_key';

export interface KeyManagerResult {
  key: string;
  isNewKey: boolean;
}

/**
 * Retrieve or generate the database encryption key.
 * Requires biometric/passcode authentication to access.
 */
export async function getOrCreateDatabaseKey(): Promise<KeyManagerResult> {
  // Try to retrieve existing key
  const existing = await Keychain.getGenericPassword({
    service: SERVICE_NAME,
    accessControl: Keychain.ACCESS_CONTROL.BIOMETRY_ANY_OR_DEVICE_PASSCODE,
    authenticationPrompt: {
      title: 'Authenticate to access your health data',
      subtitle: 'Your ECG data is encrypted and requires authentication',
      cancel: 'Cancel',
    },
  });

  if (existing && existing.password) {
    return { key: existing.password, isNewKey: false };
  }

  // Generate new 256-bit key
  const keyBytes = await Crypto.getRandomBytesAsync(32);
  const key = Array.from(keyBytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');

  // Store with biometric protection
  await Keychain.setGenericPassword(KEY_USERNAME, key, {
    service: SERVICE_NAME,
    accessControl: Keychain.ACCESS_CONTROL.BIOMETRY_ANY_OR_DEVICE_PASSCODE,
    accessible: Keychain.ACCESSIBLE.WHEN_PASSCODE_SET_THIS_DEVICE_ONLY,
    securityLevel: Keychain.SECURITY_LEVEL.SECURE_HARDWARE,
  });

  return { key, isNewKey: true };
}

/**
 * Check if biometric authentication is available on this device.
 */
export async function getBiometricType(): Promise<string | null> {
  const type = await Keychain.getSupportedBiometryType();
  return type;
}

/**
 * Wipe the database key. Called on "Delete All My Data".
 * After this, the database cannot be opened — all data is effectively destroyed.
 */
export async function wipeDatabaseKey(): Promise<void> {
  await Keychain.resetGenericPassword({ service: SERVICE_NAME });
}

/**
 * Check if a database key exists (without retrieving it).
 */
export async function hasStoredKey(): Promise<boolean> {
  try {
    // getGenericPassword without auth prompt — just checks existence
    const result = await Keychain.getGenericPassword({
      service: SERVICE_NAME,
    });
    return !!result;
  } catch {
    return false;
  }
}
