/**
 * API service for backend communication
 */

import { API_ENDPOINTS } from '@/constants/api';

export interface ChunkProcessResponse {
  chunk_index: number;
  session_id: string;
  cough_count: number;
  wheeze_windows: number;
  windows_processed: number;
  detected_events: Array<{
    start_ms: number;
    end_ms: number;
    confidence: number;
    tags: string[];
    quality_flag?: string;
    window_indices: number[];
  }>;
}

export interface NightlySummary {
  session_id: string;
  patient_id?: number;
  age?: number;
  sex?: string;
  total_duration_minutes: number;
  coughs_per_hour: number;
  bout_count: number;
  bout_lengths: number[];
  inter_cough_intervals: number[];
  avg_bout_length_seconds?: number;
  avg_inter_cough_interval_seconds?: number;
  max_coughs_in_single_hour: number;
  wheeze_time_percent: number;
  longest_wheeze_duration_seconds?: number;
  wheeze_intensity_avg?: number;
  attribute_prevalence: {
    wet: number;
    stridor: number;
    choking: number;
    congestion: number;
    selfreported_wheezing: number;
  };
  cough_events: Array<{
    start_ms: number;
    end_ms: number;
    confidence: number;
    tags: string[];
    quality_flag?: string;
    window_indices: number[];
  }>;
  pattern_scores?: Array<{
    pattern_name: string;
    score: number;
    uncertainty_lower: number;
    uncertainty_upper: number;
    why: string;
  }>;
  hourly_breakdown: Array<{
    hour: number;
    cough_count: number;
    wheeze_percent: number;
    events: any[];
  }>;
  quality_metrics: {
    avg_snr: number;
    quality_score: number;
    low_quality_periods_count: number;
    high_confidence_events_count: number;
    suppressed_events_count: number;
  };
  display_strings: {
    sleep_duration_formatted: string;
    coughs_per_hour_formatted: string;
    severity_badge_color: string;
    overall_quality_score: number;
  };
  dedalus_interpretation?: {
    interpretation: string;
    severity?: string;
    recommendations?: string[];
  };
}

/**
 * Process a 10-minute audio chunk
 */
export async function processChunk(
  audioBlob: Blob,
  chunkIndex: number,
  sessionId: string,
  patientId?: number,
  age?: number,
  sex?: string
): Promise<ChunkProcessResponse> {
  const formData = new FormData();
  formData.append('audio_chunk', audioBlob, 'chunk.wav');
  formData.append('chunk_index', chunkIndex.toString());
  formData.append('session_id', sessionId);
  if (patientId !== undefined) {
    formData.append('patient_id', patientId.toString());
  }
  if (age !== undefined) {
    formData.append('age', age.toString());
  }
  if (sex !== undefined) {
    formData.append('sex', sex);
  }

  const response = await fetch(API_ENDPOINTS.PROCESS_CHUNK, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to process chunk: ${response.status} ${errorText}`);
  }

  return response.json();
}

/**
 * Get final nightly summary
 */
export async function getFinalSummary(
  sessionId: string,
  symptomForm?: {
    fever: boolean;
    sore_throat: boolean;
    chest_tightness: boolean;
    duration: number;
    nocturnal_worsening: boolean;
    asthma_history: boolean;
    copd_history: boolean;
    age_band?: string;
    smoker: boolean;
  }
): Promise<NightlySummary> {
  const response = await fetch(API_ENDPOINTS.FINAL_SUMMARY(sessionId), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: symptomForm ? JSON.stringify({ symptom_form: symptomForm }) : undefined,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to get final summary: ${response.status} ${errorText}`);
  }

  return response.json();
}

/**
 * Check backend health status
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(API_ENDPOINTS.HEALTH);
    return response.ok;
  } catch {
    return false;
  }
}

