/**
 * Main app layout — tab navigation for authenticated users.
 *
 * Only rendered after successful biometric authentication.
 * All screens within this layout can access the encrypted database.
 */

import { Tabs } from 'expo-router';
import { Text } from 'react-native';

export default function MainLayout() {
  return (
    <Tabs
      screenOptions={{
        headerStyle: { backgroundColor: '#071312' },
        headerTintColor: '#00E5B0',
        tabBarStyle: {
          backgroundColor: '#0D1F1E',
          borderTopColor: '#1E3533',
        },
        tabBarActiveTintColor: '#00E5B0',
        tabBarInactiveTintColor: '#5A8A85',
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Dashboard',
          tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 18, fontWeight: '700' }}>H</Text>,
        }}
      />
      <Tabs.Screen
        name="scan"
        options={{
          title: 'Scan',
          tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 18, fontWeight: '700' }}>S</Text>,
        }}
      />
      <Tabs.Screen
        name="patients"
        options={{
          title: 'Patients',
          tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 18, fontWeight: '700' }}>P</Text>,
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 18, fontWeight: '700' }}>G</Text>,
        }}
      />
      {/* Hide dynamic routes from tab bar */}
      <Tabs.Screen name="analyze" options={{ href: null }} />
      <Tabs.Screen name="patient/[id]" options={{ href: null }} />
      <Tabs.Screen name="patient/new" options={{ href: null }} />
    </Tabs>
  );
}
