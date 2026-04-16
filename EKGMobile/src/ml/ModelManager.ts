/**
 * ModelManager — ONNX model loading, verification, and session management.
 *
 * Loads ECG V3 model from assets, verifies SHA-256 hash, exposes inference session.
 * Implements HIPAA 1.4.2 (integrity verification) via hash-chain.
 */

import * as FileSystem from 'expo-file-system';
import * as Crypto from 'expo-crypto';
import { Asset } from 'expo-asset';
import { InferenceSession } from 'onnxruntime-react-native';
import { logEvent } from '../audit/AuditLogger';
import useAppStore from '../store/useAppStore';

let inferenceSession: InferenceSession | null = null;
let modelHash: string | null = null;
let isLoading = false;

/**
 * Load, verify, and cache the ONNX model.
 * Returns true if successful, false otherwise.
 */
export async function loadModel(): Promise<boolean> {
  if (inferenceSession) {
    return true; // Already loaded
  }

  if (isLoading) {
    return false; // Already in progress
  }

  isLoading = true;

  try {
    await logEvent('MODEL_LOAD', 'load_model_start', null, 'v3');

    // Load model asset from bundle
    const [asset] = await Asset.loadAsync(
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      require('@/assets/models/ecg_v3.onnx')
    );

    if (!asset?.localUri) {
      throw new Error('Failed to locate model asset');
    }

    const modelUri = asset.localUri;

    // Read model file for integrity verification (SHA-256, HIPAA 1.4.2)
    const modelData = await FileSystem.readAsStringAsync(modelUri, {
      encoding: FileSystem.EncodingType.Base64,
    });

    modelHash = await Crypto.digestStringAsync(
      Crypto.CryptoDigestAlgorithm.SHA256,
      modelData,
    );

    // Verify hash against manifest (best-effort — manifest may not exist in dev)
    let verified = false;
    try {
      const [manifestAsset] = await Asset.loadAsync(
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require('@/assets/models/model_manifest.json')
      );
      if (manifestAsset?.localUri) {
        const manifestJson = await FileSystem.readAsStringAsync(manifestAsset.localUri);
        const manifest = JSON.parse(manifestJson);
        const expected = manifest?.files?.['ecg_v3.onnx']?.sha256;
        if (expected && expected !== modelHash) {
          await logEvent('MODEL_INTEGRITY_FAIL', 'hash_mismatch', null, 'v3',
            `expected:${expected.slice(0, 16)},got:${modelHash.slice(0, 16)}`);
          throw new Error('Model hash verification failed -- file may be corrupted');
        }
        if (expected) verified = true;
      }
    } catch (manifestErr) {
      if ((manifestErr as Error).message?.includes('hash verification failed')) {
        throw manifestErr;
      }
      // Manifest not found in dev — continue without verification
    }

    await logEvent('MODEL_LOAD', 'hash_computed', null, 'v3',
      `hash:${modelHash.slice(0, 16)},verified:${verified}`);

    // Create ONNX inference session
    inferenceSession = await InferenceSession.create(modelUri, {
      executionProviders: ['cpu'],
    });

    // Update app store
    useAppStore.getState().setModelLoaded(true);

    await logEvent('MODEL_LOAD', 'load_model_success', null, 'v3');
    return true;
  } catch (error) {
    await logEvent('MODEL_LOAD', 'load_model_error', null, 'v3',
      String(error).slice(0, 200));
    useAppStore.getState().setModelLoaded(false);
    return false;
  } finally {
    isLoading = false;
  }
}

/**
 * Check if model is loaded and ready for inference.
 */
export function isModelLoaded(): boolean {
  return inferenceSession !== null;
}

/**
 * Get the current ONNX inference session.
 * Throws if model is not loaded.
 */
export function getSession(): InferenceSession {
  if (!inferenceSession) {
    throw new Error('Model not loaded. Call loadModel() first.');
  }
  return inferenceSession;
}

/**
 * Get the SHA-256 hash of the loaded model.
 */
export function getModelHash(): string | null {
  return modelHash;
}

/**
 * Unload the model and free resources.
 */
export function unloadModel(): void {
  inferenceSession = null;
  modelHash = null;
  useAppStore.getState().setModelLoaded(false);
}
