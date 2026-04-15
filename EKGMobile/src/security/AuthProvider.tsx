/**
 * AuthProvider — HIPAA 1.2.1: Biometric authentication gate.
 *
 * Wraps the entire app. No PHI-containing screens are rendered until
 * the user authenticates via biometrics (FaceID/TouchID/fingerprint)
 * or device PIN/passcode.
 *
 * Integrates with:
 * - KeyManager (retrieve DB encryption key on auth success)
 * - SessionManager (auto-lock after inactivity)
 * - ScreenProtection (FLAG_SECURE + blur on background)
 * - AuditLogger (log auth events)
 */

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import * as LocalAuthentication from 'expo-local-authentication';
import { getOrCreateDatabaseKey, getBiometricType } from './KeyManager';
import { enableScreenProtection, setLifecycleCallbacks } from './ScreenProtection';
import * as SessionManager from './SessionManager';
import { logEvent } from '../audit/AuditLogger';

interface AuthContextValue {
  isAuthenticated: boolean;
  isLoading: boolean;
  databaseKey: string | null;
  lock: () => void;
  authenticate: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextValue>({
  isAuthenticated: false,
  isLoading: true,
  databaseKey: null,
  lock: () => {},
  authenticate: async () => false,
});

export function useAuth(): AuthContextValue {
  return useContext(AuthContext);
}

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps): React.JSX.Element {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [databaseKey, setDatabaseKey] = useState<string | null>(null);
  const [authError, setAuthError] = useState<string | null>(null);
  const [biometricType, setBiometricType] = useState<string | null>(null);
  const touchRef = useRef<ReturnType<typeof setTimeout>>();

  // Lock the app (clear auth state and DB key)
  const lock = useCallback(() => {
    setIsAuthenticated(false);
    setDatabaseKey(null);
    logEvent('AUTH_LOCK', 'session_locked').catch(() => {});
  }, []);

  // Authenticate the user
  const authenticate = useCallback(async (): Promise<boolean> => {
    setIsLoading(true);
    setAuthError(null);

    try {
      // DEV mode: skip biometrics (no hardware in Expo Go / browser)
      if (__DEV__) {
        console.warn('[AUTH] DEV MODE — skipping biometric, using dummy DB key');
        setDatabaseKey('0'.repeat(64));
        setIsAuthenticated(true);
        setIsLoading(false);
        return true;
      }

      // Check hardware biometric support
      const hasHardware = await LocalAuthentication.hasHardwareAsync();
      const isEnrolled = await LocalAuthentication.isEnrolledAsync();

      if (!hasHardware || !isEnrolled) {
        setAuthError(
          'This device does not have biometric authentication set up. ' +
          'Please enable Face ID, Touch ID, or fingerprint in your device settings.'
        );
        await logEvent('AUTH_FAIL', 'no_biometrics_available');
        setIsLoading(false);
        return false;
      }

      // Prompt biometric
      const result = await LocalAuthentication.authenticateAsync({
        promptMessage: 'Authenticate to access your health data',
        cancelLabel: 'Cancel',
        fallbackLabel: 'Use Passcode',
        disableDeviceFallback: false,
      });

      if (result.success) {
        // Retrieve database key from hardware keystore
        const { key } = await getOrCreateDatabaseKey();
        setDatabaseKey(key);
        setIsAuthenticated(true);

        // Enable screen protection after successful auth
        enableScreenProtection();

        // Start inactivity timer
        SessionManager.startSession(lock);

        await logEvent('AUTH_SUCCESS', 'biometric_auth');
        setIsLoading(false);
        return true;
      } else {
        setAuthError('Authentication failed. Please try again.');
        await logEvent('AUTH_FAIL', 'biometric_rejected');
        setIsLoading(false);
        return false;
      }
    } catch (error) {
      setAuthError('Authentication error. Please try again.');
      await logEvent('AUTH_FAIL', 'auth_error', null, null, 'exception_during_auth');
      setIsLoading(false);
      return false;
    }
  }, [lock]);

  // Setup lifecycle callbacks for background/foreground
  useEffect(() => {
    setLifecycleCallbacks(
      // On background: lock the app
      () => {
        if (isAuthenticated) {
          lock();
        }
      },
      // On foreground: trigger re-auth (handled by lock screen rendering)
      () => {},
    );
  }, [isAuthenticated, lock]);

  // Reset inactivity timer on any touch
  const handleTouch = useCallback(() => {
    SessionManager.resetTimer();
  }, []);

  // Check biometric type on mount
  useEffect(() => {
    getBiometricType().then(setBiometricType);
    setIsLoading(false);
  }, []);

  // If not authenticated, show lock screen
  if (!isAuthenticated) {
    return (
      <View style={styles.lockScreen} onTouchStart={handleTouch}>
        <View style={styles.lockContent}>
          <Text style={styles.lockIcon}>🔒</Text>
          <Text style={styles.lockTitle}>EKG Intelligence</Text>
          <Text style={styles.lockSubtitle}>
            Your health data is encrypted and protected
          </Text>

          {authError && (
            <Text style={styles.errorText}>{authError}</Text>
          )}

          {isLoading ? (
            <ActivityIndicator size="large" color="#00E5B0" style={styles.loader} />
          ) : (
            <TouchableOpacity
              style={styles.authButton}
              onPress={authenticate}
              activeOpacity={0.7}
            >
              <Text style={styles.authButtonText}>
                {biometricType
                  ? `Authenticate with ${biometricType}`
                  : 'Authenticate'}
              </Text>
            </TouchableOpacity>
          )}

          <Text style={styles.disclaimer}>
            For educational purposes only. Not FDA-cleared.
          </Text>
        </View>
      </View>
    );
  }

  // Authenticated — render the app with touch tracking for inactivity
  return (
    <AuthContext.Provider
      value={{ isAuthenticated, isLoading, databaseKey, lock, authenticate }}
    >
      <View style={styles.container} onTouchStart={handleTouch}>
        {children}
      </View>
    </AuthContext.Provider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  lockScreen: {
    flex: 1,
    backgroundColor: '#071312',
    justifyContent: 'center',
    alignItems: 'center',
  },
  lockContent: {
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  lockIcon: {
    fontSize: 48,
    marginBottom: 16,
  },
  lockTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: '#00E5B0',
    marginBottom: 8,
  },
  lockSubtitle: {
    fontSize: 14,
    color: '#5A8A85',
    textAlign: 'center',
    marginBottom: 32,
  },
  errorText: {
    fontSize: 14,
    color: '#FF6B6B',
    textAlign: 'center',
    marginBottom: 16,
  },
  loader: {
    marginVertical: 24,
  },
  authButton: {
    backgroundColor: '#00E5B0',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
    marginBottom: 32,
  },
  authButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#071312',
  },
  disclaimer: {
    fontSize: 11,
    color: '#3D6662',
    textAlign: 'center',
    position: 'absolute',
    bottom: -80,
  },
});
