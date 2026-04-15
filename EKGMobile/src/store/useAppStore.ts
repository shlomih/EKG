/**
 * Global app state store using Zustand
 * Manages authentication, patient context, model, and settings
 * IMPORTANT: No PHI stored here — patient names/IDs/ECG data stay in encrypted DB only
 */
import { create } from 'zustand';

interface AppState {
  // Auth state
  isAuthenticated: boolean;
  lastActivity: number;
  setAuthenticated: (value: boolean) => void;
  updateLastActivity: () => void;

  // Current patient context (reference pointer only)
  currentPatientId: string | null;
  setCurrentPatient: (id: string | null) => void;

  // Model info
  modelVersion: string;
  modelLoaded: boolean;
  setModelLoaded: (loaded: boolean) => void;

  // Settings
  language: string;
  setLanguage: (lang: string) => void;
  inactivityTimeoutMs: number;
  setInactivityTimeout: (ms: number) => void;
}

const useAppStore = create<AppState>((set) => ({
  // Auth state
  isAuthenticated: false,
  lastActivity: Date.now(),
  setAuthenticated: (value) => set({ isAuthenticated: value }),
  updateLastActivity: () => set({ lastActivity: Date.now() }),

  // Current patient context
  currentPatientId: null,
  setCurrentPatient: (id) => set({ currentPatientId: id }),

  // Model info
  modelVersion: 'V3',
  modelLoaded: false,
  setModelLoaded: (loaded) => set({ modelLoaded: loaded }),

  // Settings
  language: 'en',
  setLanguage: (lang) => set({ language: lang }),
  inactivityTimeoutMs: 300000, // 5 minutes default
  setInactivityTimeout: (ms) => set({ inactivityTimeoutMs: ms }),
}));

export default useAppStore;
