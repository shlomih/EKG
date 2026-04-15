/**
 * ScreenProtection — HIPAA 1.2.3: Prevent PHI leakage via screenshots
 * and app switcher previews.
 *
 * Android: FLAG_SECURE prevents screenshots and hides content in recents.
 * iOS: Overlay blur on app backgrounding to hide PHI in app switcher.
 */

import { AppState, AppStateStatus, Platform } from 'react-native';
import * as ScreenCapture from 'expo-screen-capture';

let appStateSubscription: ReturnType<typeof AppState.addEventListener> | null = null;
let isProtectionActive = false;

/**
 * Enable screen protection. Call once on app startup after authentication.
 * - Android: Activates FLAG_SECURE (blocks screenshots + recents preview)
 * - iOS: Registers listener to blur content on app backgrounding
 */
export function enableScreenProtection(): void {
  if (isProtectionActive) return;

  if (Platform.OS === 'android') {
    // FLAG_SECURE: screenshots return blank, recents show blank thumbnail
    ScreenCapture.preventScreenCaptureAsync('hipaa_protection');
  }

  // Both platforms: listen for app state changes
  appStateSubscription = AppState.addEventListener('change', handleAppStateChange);
  isProtectionActive = true;
}

/**
 * Disable screen protection. Called during teardown.
 */
export function disableScreenProtection(): void {
  if (!isProtectionActive) return;

  if (Platform.OS === 'android') {
    ScreenCapture.allowScreenCaptureAsync('hipaa_protection');
  }

  if (appStateSubscription) {
    appStateSubscription.remove();
    appStateSubscription = null;
  }
  isProtectionActive = false;
}

/**
 * Handle app state transitions. When app goes to background or inactive,
 * sensitive content should be hidden.
 */
function handleAppStateChange(nextState: AppStateStatus): void {
  if (nextState === 'active') {
    // App returned to foreground — content will be revealed after re-auth
    onAppForeground?.();
  } else {
    // App going to background/inactive — hide content
    onAppBackground?.();
  }
}

// Callbacks for auth provider to wire up lock/unlock behavior
let onAppBackground: (() => void) | null = null;
let onAppForeground: (() => void) | null = null;

/**
 * Register callbacks for app lifecycle events.
 * AuthProvider uses these to trigger lock screen on background.
 */
export function setLifecycleCallbacks(
  onBackground: () => void,
  onForeground: () => void,
): void {
  onAppBackground = onBackground;
  onAppForeground = onForeground;
}
