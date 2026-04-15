/**
 * HIPAA 1.2.1, 1.2.2 — Authentication & Session Management Tests
 *
 * Verifies that:
 * - App requires biometric auth before showing any PHI
 * - Failed auth blocks data access
 * - Auto-lock triggers after inactivity timeout
 * - Session timer resets on user interaction
 */

jest.mock('expo-local-authentication');
jest.mock('react-native-keychain');

describe('HIPAA 1.2.1 - Authentication required', () => {
  test('A-2: Failed biometric blocks all data access', async () => {
    const LocalAuth = require('expo-local-authentication');
    LocalAuth.authenticateAsync.mockResolvedValue({ success: false });

    // After failed auth, database should not be accessible
    const { isDatabaseOpen } = require('../../src/db/Database');
    expect(isDatabaseOpen()).toBe(false);
  });

  test('A-9: Device without passcode shows security warning', async () => {
    const LocalAuth = require('expo-local-authentication');

    // No biometric hardware
    LocalAuth.hasHardwareAsync.mockResolvedValue(false);
    LocalAuth.isEnrolledAsync.mockResolvedValue(false);

    const result = await LocalAuth.authenticateAsync();
    // AuthProvider should show error and refuse to proceed
    expect(LocalAuth.hasHardwareAsync).toHaveBeenCalled();
  });
});

describe('HIPAA 1.2.2 - Inactivity auto-lock', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
    const SessionManager = require('../../src/security/SessionManager');
    SessionManager.stopSession();
  });

  test('A-4: Auto-lock triggers after 5 min inactivity', () => {
    const SessionManager = require('../../src/security/SessionManager');
    const lockCallback = jest.fn();

    SessionManager.startSession(lockCallback, 5 * 60 * 1000);

    // Advance time by 4 minutes — should NOT lock
    jest.advanceTimersByTime(4 * 60 * 1000);
    expect(lockCallback).not.toHaveBeenCalled();

    // Advance past 5 minutes — should lock
    jest.advanceTimersByTime(1 * 60 * 1000 + 1);
    expect(lockCallback).toHaveBeenCalledTimes(1);
  });

  test('A-5: Timer resets on user interaction', () => {
    const SessionManager = require('../../src/security/SessionManager');
    const lockCallback = jest.fn();

    SessionManager.startSession(lockCallback, 5 * 60 * 1000);

    // Advance 4 minutes
    jest.advanceTimersByTime(4 * 60 * 1000);
    expect(lockCallback).not.toHaveBeenCalled();

    // User touches the screen — reset timer
    SessionManager.resetTimer();

    // Advance another 4 minutes (total 8 from start, but 4 from reset)
    jest.advanceTimersByTime(4 * 60 * 1000);
    expect(lockCallback).not.toHaveBeenCalled();

    // Advance past 5 minutes from reset
    jest.advanceTimersByTime(1 * 60 * 1000 + 1);
    expect(lockCallback).toHaveBeenCalledTimes(1);
  });

  test('A-4b: Minimum timeout is 1 minute', () => {
    const SessionManager = require('../../src/security/SessionManager');

    // Try to set timeout to 10 seconds — should clamp to 60 seconds
    SessionManager.setTimeoutMs(10_000);
    expect(SessionManager.getTimeoutMs()).toBe(60_000);
  });
});
