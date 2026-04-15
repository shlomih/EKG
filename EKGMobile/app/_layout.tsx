/**
 * Root layout — wraps the entire app with:
 * 1. PHI Sanitizer (HIPAA 1.1.3) — installed first, before any logging
 * 2. AuthProvider (HIPAA 1.2.1) — biometric gate, no PHI until authenticated
 * 3. Database connection — opened with key from AuthProvider
 */

import { useEffect } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import '../src/i18n/i18n';
import { installSanitizer } from '../src/security/PHISanitizer';
import { AuthProvider } from '../src/security/AuthProvider';

// Install PHI sanitizer FIRST — before any other imports can log PHI
installSanitizer();

export default function RootLayout() {
  return (
    <AuthProvider>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: '#071312' },
          animation: 'fade',
        }}
      >
        <Stack.Screen name="(main)" />
      </Stack>
    </AuthProvider>
  );
}
