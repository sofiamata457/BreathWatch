/**
 * Utility to convert mock weekly data into NightlySummary format for history/statistics
 */

import { MOCK_WEEKLY_DATA, DailyData } from './mockWeeklyData';
import { NightlySummary } from '@/services/api';
import { RecordingHistoryItem } from '@/services/storage';

/**
 * Convert a DailyData entry into a NightlySummary
 */
function dailyDataToNightlySummary(dayData: DailyData, index: number): NightlySummary {
  const sessionId = `mock_session_${dayData.fullDate.replace(/-/g, '')}`;
  const date = new Date(dayData.fullDate);
  
  // Create mock events based on cough count
  const coughEvents = Array.from({ length: dayData.coughCount }, (_, i) => ({
    start_ms: i * 5000, // 5 seconds apart
    end_ms: (i * 5000) + 1000, // 1 second duration
    confidence: 0.7 + Math.random() * 0.3, // 0.7-1.0
    tags: [] as string[],
    window_indices: [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3], // Mock window indices
  }));

  // Create mock probability timeline
  const numWindows = 240; // 1 hour of 1-second windows (for 4 hours of sleep)
  const timeline = {
    tile_seconds: 1.0,
    stride_seconds: 0.25,
    indices: Array.from({ length: numWindows }, (_, i) => i),
    times: Array.from({ length: numWindows }, (_, i) => i),
    p_cough: Array.from({ length: numWindows }, (_, i) => {
      // Higher probability during cough events
      const inEvent = coughEvents.some(
        (event) => i * 0.25 >= event.start_ms / 1000 && i * 0.25 <= event.end_ms / 1000
      );
      return inEvent ? 0.6 + Math.random() * 0.4 : Math.random() * 0.3;
    }),
    attr_series: {
      wet: Array.from({ length: numWindows }, () => dayData.attributePrevalence.wet / 100),
      wheezing: Array.from({ length: numWindows }, () => dayData.attributePrevalence.wheezing / 100),
      stridor: Array.from({ length: numWindows }, () => dayData.attributePrevalence.stridor / 100),
      choking: Array.from({ length: numWindows }, () => dayData.attributePrevalence.choking / 100),
      congestion: Array.from({ length: numWindows }, () => dayData.attributePrevalence.congestion / 100),
    },
  };

  // Create mock event summary
  const eventSummary = {
    num_events: dayData.coughCount,
    events: coughEvents.map((event, idx) => ({
      start: event.start_ms / 1000,
      end: event.end_ms / 1000,
      duration: (event.end_ms - event.start_ms) / 1000,
      tile_indices: event.window_indices,
      p_cough_max: 0.8 + Math.random() * 0.2,
      p_cough_mean: 0.7 + Math.random() * 0.2,
      attr_probs: {
        wet: dayData.attributePrevalence.wet / 100,
        wheezing: dayData.attributePrevalence.wheezing / 100,
        stridor: dayData.attributePrevalence.stridor / 100,
        choking: dayData.attributePrevalence.choking / 100,
        congestion: dayData.attributePrevalence.congestion / 100,
      },
      attr_flags: {
        wet: dayData.attributePrevalence.wet > 50 ? 1 : 0,
        wheezing: dayData.attributePrevalence.wheezing > 50 ? 1 : 0,
        stridor: dayData.attributePrevalence.stridor > 50 ? 1 : 0,
        choking: dayData.attributePrevalence.choking > 50 ? 1 : 0,
        congestion: dayData.attributePrevalence.congestion > 50 ? 1 : 0,
      },
    })),
  };

  // Determine severity based on cough count
  let severity = 'Normal';
  let severityColor = '#4caf50';
  if (dayData.coughCount > 12) {
    severity = 'Severe';
    severityColor = '#ff4444';
  } else if (dayData.coughCount > 7) {
    severity = 'Moderate';
    severityColor = '#ff9800';
  } else if (dayData.coughCount > 3) {
    severity = 'Mild';
    severityColor = '#ffc107';
  }

  return {
    session_id: sessionId,
    total_duration_minutes: 240, // 4 hours
    coughs_per_hour: dayData.coughsPerHour,
    bout_count: Math.ceil(dayData.coughCount / 3), // Rough estimate
    bout_lengths: Array.from({ length: Math.ceil(dayData.coughCount / 3) }, () => 3),
    inter_cough_intervals: Array.from({ length: dayData.coughCount - 1 }, () => 30),
    avg_bout_length_seconds: 3.0,
    avg_inter_cough_interval_seconds: 30.0,
    max_coughs_in_single_hour: dayData.coughCount,
    wheeze_time_percent: dayData.wheezeTimePercent,
    longest_wheeze_duration_seconds: dayData.wheezeTimePercent > 0 ? 60 : undefined,
    wheeze_intensity_avg: dayData.wheezeTimePercent > 0 ? dayData.attributePrevalence.wheezing / 100 : undefined,
    attribute_prevalence: dayData.attributePrevalence,
    cough_events: coughEvents,
    event_summary,
    probability_timeline: timeline,
    hourly_breakdown: [
      {
        hour: 0,
        cough_count: Math.floor(dayData.coughCount * 0.3),
        wheeze_percent: dayData.wheezeTimePercent * 0.3,
        events: [],
      },
      {
        hour: 1,
        cough_count: Math.floor(dayData.coughCount * 0.4),
        wheeze_percent: dayData.wheezeTimePercent * 0.4,
        events: [],
      },
      {
        hour: 2,
        cough_count: Math.floor(dayData.coughCount * 0.2),
        wheeze_percent: dayData.wheezeTimePercent * 0.2,
        events: [],
      },
      {
        hour: 3,
        cough_count: Math.floor(dayData.coughCount * 0.1),
        wheeze_percent: dayData.wheezeTimePercent * 0.1,
        events: [],
      },
    ],
    quality_metrics: {
      avg_snr: 15 + Math.random() * 10, // 15-25 dB
      quality_score: 70 + Math.random() * 25, // 70-95
      low_quality_periods_count: Math.floor(Math.random() * 3),
      high_confidence_events_count: Math.floor(dayData.coughCount * 0.8),
      suppressed_events_count: 0,
    },
    display_strings: {
      sleep_duration_formatted: '4h 0m',
      coughs_per_hour_formatted: `${dayData.coughsPerHour.toFixed(1)}/hr`,
      severity_badge_color: severityColor,
      overall_quality_score: 75 + Math.random() * 20,
    },
  };
}

/**
 * Generate mock history items from weekly data
 */
export function generateMockHistoryItems(): RecordingHistoryItem[] {
  return MOCK_WEEKLY_DATA.map((dayData, index) => {
    const summary = dailyDataToNightlySummary(dayData, index);
    const date = new Date(dayData.fullDate);
    // Set time to evening (10 PM) for each day
    date.setHours(22, 0, 0, 0);
    
    return {
      sessionId: summary.session_id,
      summary,
      timestamp: date.getTime(),
      date: date.toISOString(),
    };
  });
}

/**
 * Get history with mock data included (if no real data exists)
 */
export async function getHistoryWithMockData(): Promise<RecordingHistoryItem[]> {
  const { getRecordingHistory } = await import('@/services/storage');
  const realHistory = await getRecordingHistory();
  
  // If there's real data, return it
  if (realHistory.length > 0) {
    return realHistory;
  }
  
  // Otherwise, return mock data
  return generateMockHistoryItems();
}

