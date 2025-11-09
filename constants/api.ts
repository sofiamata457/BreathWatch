/**
 * API configuration constants
 */

// Backend API URL - can be overridden with environment variable
export const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  PROCESS_CHUNK: `${API_BASE_URL}/process-chunk`,
  FINAL_SUMMARY: (sessionId: string) => `${API_BASE_URL}/final-summary/${sessionId}`,
  STATUS: `${API_BASE_URL}/status`,
  HEALTH: `${API_BASE_URL}/`,
} as const;


