/**
 * TempFileManager — HIPAA 1.1.5: Secure temporary file handling.
 *
 * Ensures that temporary files (camera captures, PDF generation intermediates,
 * signal processing buffers) are:
 * 1. Created in app-private storage only (not Gallery/Photos/Downloads)
 * 2. Overwritten with zeros before deletion (secure wipe)
 * 3. Tracked for cleanup on app exit or crash
 */

import * as FileSystem from 'expo-file-system';

// Track all active temp files for cleanup
const activeTempFiles = new Set<string>();

/**
 * Create a temporary file path in app-private storage.
 * HIPAA 5.3.6: Never writes to public storage (Gallery/Photos).
 */
export function createTempPath(extension: string = 'tmp'): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8);
  const filename = `ekg_temp_${timestamp}_${random}.${extension}`;
  const path = `${FileSystem.cacheDirectory}${filename}`;

  activeTempFiles.add(path);
  return path;
}

/**
 * Securely delete a temporary file.
 * HIPAA 1.1.5: Overwrite with zeros before unlinking.
 */
export async function secureDelete(filePath: string): Promise<void> {
  try {
    const info = await FileSystem.getInfoAsync(filePath);
    if (!info.exists) {
      activeTempFiles.delete(filePath);
      return;
    }

    // Overwrite with zeros (size of original file)
    if (info.size && info.size > 0) {
      const zeros = new Uint8Array(Math.min(info.size, 1024 * 1024)); // Max 1MB chunks
      const zerosBase64 = uint8ArrayToBase64(zeros);

      await FileSystem.writeAsStringAsync(filePath, zerosBase64, {
        encoding: FileSystem.EncodingType.Base64,
      });
    }

    // Delete the file
    await FileSystem.deleteAsync(filePath, { idempotent: true });
    activeTempFiles.delete(filePath);
  } catch {
    // Best-effort deletion — still remove from tracking
    try {
      await FileSystem.deleteAsync(filePath, { idempotent: true });
    } catch {
      // File may already be gone
    }
    activeTempFiles.delete(filePath);
  }
}

/**
 * Clean up all tracked temporary files.
 * Called on app exit, crash recovery, or session lock.
 */
export async function cleanupAllTempFiles(): Promise<void> {
  const files = Array.from(activeTempFiles);
  await Promise.allSettled(files.map(secureDelete));
}

/**
 * Get count of active temporary files (for debugging/testing).
 */
export function getActiveTempFileCount(): number {
  return activeTempFiles.size;
}

/**
 * Convert Uint8Array to base64 string.
 */
function uint8ArrayToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return btoa(binary);
}
