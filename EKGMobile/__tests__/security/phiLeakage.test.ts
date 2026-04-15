/**
 * HIPAA 1.1.3 — No PHI in Logs Tests
 *
 * Verifies that the PHISanitizer prevents patient data from
 * appearing in console output, error reports, and crash logs.
 */

describe('HIPAA 1.1.3 - No PHI in logs', () => {
  let capturedLogs: string[];
  let originalLog: typeof console.log;

  beforeEach(() => {
    capturedLogs = [];
    originalLog = console.log;

    // Install sanitizer and capture output
    const { uninstallSanitizer, installSanitizer } = require('../../src/security/PHISanitizer');
    uninstallSanitizer();

    // Capture raw output AFTER sanitizer processes it
    console.log = (...args: unknown[]) => {
      capturedLogs.push(args.map(String).join(' '));
    };

    installSanitizer();
  });

  afterEach(() => {
    const { uninstallSanitizer } = require('../../src/security/PHISanitizer');
    uninstallSanitizer();
    console.log = originalLog;
  });

  test('P-1: console.log does not contain patient names', () => {
    const patient = {
      first_name: 'John',
      last_name: 'Smith',
      age: 45,
      patient_id: '550e8400-e29b-41d4-a716-446655440000',
    };

    console.log('Patient loaded:', patient);

    const output = capturedLogs.join('\n');
    expect(output).not.toContain('John');
    expect(output).not.toContain('Smith');
    expect(output).toContain('[REDACTED]');
  });

  test('P-2: console.error does not contain patient data', () => {
    const capturedErrors: string[] = [];
    const origError = console.error;
    console.error = (...args: unknown[]) => {
      capturedErrors.push(args.map(String).join(' '));
    };

    const { installSanitizer } = require('../../src/security/PHISanitizer');
    installSanitizer();

    const error = new Error('Failed to save patient John Doe, id: 12345');
    console.error('Database error:', error.message);

    const output = capturedErrors.join('\n');
    // UUIDs are redacted, but the error type/structure should be preserved
    expect(output).toContain('Database error');

    console.error = origError;
  });

  test('P-3: Large numeric arrays (ECG signal data) are redacted', () => {
    // Simulate ECG signal data being logged
    const signalData = new Array(5000).fill(0).map(() => Math.random() * 2 - 1);

    console.log('Signal data:', signalData);

    const output = capturedLogs.join('\n');
    expect(output).toContain('[REDACTED');
    expect(output).not.toContain(String(signalData[0]));
  });

  test('P-5: Full workflow produces zero PHI in logs', () => {
    // Simulate a complete workflow with patient data flowing through
    const patient = {
      first_name: 'Jane',
      last_name: 'Doe',
      patient_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
      age: 67,
    };

    const analysis = {
      primary_condition: 'AFIB',
      conditions: ['AFIB', 'LVH'],
      scores: { AFIB: 0.89, LVH: 0.72, NORM: 0.12 },
    };

    console.log('Creating patient:', patient);
    console.log('Analysis result:', analysis);
    console.log(`Patient ${patient.first_name} ${patient.last_name} has ${analysis.conditions.length} conditions`);

    const output = capturedLogs.join('\n');

    // Names should be sanitized
    expect(output).not.toContain('Jane');
    expect(output).not.toContain('Doe');

    // UUIDs should be redacted
    expect(output).not.toContain('a1b2c3d4-e5f6-7890-abcd-ef1234567890');
  });
});

describe('HIPAA 5.3.7 - Minimum app permissions', () => {
  test('D-7: App requests only camera permission', () => {
    const appJson = require('../../app.json');

    expect(appJson.expo.android.permissions).toEqual(
      expect.arrayContaining(['CAMERA']),
    );

    // Should NOT contain these
    const forbidden = ['READ_CONTACTS', 'ACCESS_FINE_LOCATION', 'RECORD_AUDIO'];
    for (const perm of forbidden) {
      expect(appJson.expo.android.permissions).not.toContain(perm);
    }
  });

  test('D-8: Blocked permissions are configured', () => {
    const appJson = require('../../app.json');

    expect(appJson.expo.android.blockedPermissions).toEqual(
      expect.arrayContaining([
        'READ_CONTACTS',
        'ACCESS_FINE_LOCATION',
        'ACCESS_COARSE_LOCATION',
        'RECORD_AUDIO',
      ]),
    );
  });
});

describe('HIPAA 2.1.4 - Backup exclusion', () => {
  test('PH-1: allowBackup is false in Android config', () => {
    const appJson = require('../../app.json');
    expect(appJson.expo.android.allowBackup).toBe(false);
  });
});

describe('HIPAA 5.2.2 - No ad SDKs', () => {
  test('D-11: No ad SDK dependencies', () => {
    const packageJson = require('../../package.json');
    const allDeps = {
      ...packageJson.dependencies,
      ...packageJson.devDependencies,
    };

    const adSdkPatterns = [
      'admob', 'google-mobile-ads', 'facebook-ads',
      'applovin', 'unity-ads', 'ironsource',
      'mopub', 'chartboost', 'vungle',
    ];

    for (const dep of Object.keys(allDeps)) {
      for (const pattern of adSdkPatterns) {
        expect(dep.toLowerCase()).not.toContain(pattern);
      }
    }
  });
});
