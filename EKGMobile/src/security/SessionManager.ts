/**
 * SessionManager — HIPAA 1.2.2: Auto-lock after inactivity timeout.
 *
 * Tracks user interaction events (touches, navigation). If no interaction
 * occurs within the configured timeout (default 5 minutes), the session
 * is locked and biometric re-authentication is required.
 */

const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

type LockCallback = () => void;

let timeoutMs = DEFAULT_TIMEOUT_MS;
let timer: ReturnType<typeof setTimeout> | null = null;
let onLock: LockCallback | null = null;
let isActive = false;

/**
 * Start the session manager with a lock callback.
 * The callback fires when the inactivity timeout expires.
 */
export function startSession(lockCallback: LockCallback, customTimeoutMs?: number): void {
  onLock = lockCallback;
  if (customTimeoutMs !== undefined) {
    timeoutMs = customTimeoutMs;
  }
  isActive = true;
  resetTimer();
}

/**
 * Stop the session manager. Called on app teardown or explicit logout.
 */
export function stopSession(): void {
  isActive = false;
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
  onLock = null;
}

/**
 * Reset the inactivity timer. Call this on every user interaction
 * (touch, scroll, navigation, button press).
 */
export function resetTimer(): void {
  if (!isActive) return;

  if (timer) {
    clearTimeout(timer);
  }

  timer = setTimeout(() => {
    if (isActive && onLock) {
      onLock();
    }
  }, timeoutMs);
}

/**
 * Get the current timeout value in milliseconds.
 */
export function getTimeoutMs(): number {
  return timeoutMs;
}

/**
 * Update the timeout value. Takes effect on next timer reset.
 */
export function setTimeoutMs(ms: number): void {
  timeoutMs = Math.max(60_000, ms); // Minimum 1 minute
}

/**
 * Check if the session manager is currently active.
 */
export function isSessionActive(): boolean {
  return isActive;
}
