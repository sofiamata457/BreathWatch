/**
 * Mock weekly data for display purposes
 * Contains 7 days of cough counts and attribute percentages
 */

export interface DailyData {
  date: string; // e.g., "Mon", "Tue", etc.
  fullDate: string; // e.g., "2025-01-13"
  coughCount: number; // Total number of coughs (not percentage)
  attributePrevalence: {
    wet: number; // Percentage (0-100)
    choking: number;
    congestion: number;
    stridor: number;
    wheezing: number;
  };
  coughsPerHour: number;
  wheezeTimePercent: number;
}

/**
 * Generate mock weekly data for the past 7 days
 */
export function generateMockWeeklyData(): DailyData[] {
  const today = new Date();
  const daysOfWeek = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  
  // Generate realistic-looking data with some variation
  const baseData: DailyData[] = [];
  
  for (let i = 6; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    const dayOfWeek = daysOfWeek[date.getDay()];
    
    // Generate varied but realistic data
    // Cough counts: 0-15 per day
    const coughCount = Math.floor(Math.random() * 16);
    
    // Attribute percentages: 0-100%, with some correlation to cough count
    const baseWet = coughCount > 0 ? Math.min(100, (coughCount * 8) + Math.random() * 20) : 0;
    const baseChoking = coughCount > 0 ? Math.min(100, (coughCount * 5) + Math.random() * 15) : 0;
    const baseCongestion = coughCount > 0 ? Math.min(100, (coughCount * 6) + Math.random() * 18) : 0;
    const baseStridor = coughCount > 0 ? Math.min(100, (coughCount * 4) + Math.random() * 12) : 0;
    const baseWheezing = coughCount > 0 ? Math.min(100, (coughCount * 7) + Math.random() * 16) : 0;
    
    baseData.push({
      date: dayOfWeek,
      fullDate: date.toISOString().split('T')[0],
      coughCount,
      attributePrevalence: {
        wet: Math.round(baseWet * 10) / 10,
        choking: Math.round(baseChoking * 10) / 10,
        congestion: Math.round(baseCongestion * 10) / 10,
        stridor: Math.round(baseStridor * 10) / 10,
        wheezing: Math.round(baseWheezing * 10) / 10,
      },
      coughsPerHour: coughCount > 0 ? Math.round((coughCount / 8) * 10) / 10 : 0, // Assuming 8 hours of sleep
      wheezeTimePercent: coughCount > 0 ? Math.round((baseWheezing / 100) * 15 * 10) / 10 : 0, // Some correlation with wheezing attribute
    });
  }
  
  return baseData;
}

/**
 * Predefined mock data for consistent display
 * This ensures the same data is shown every time for demos
 */
export const MOCK_WEEKLY_DATA: DailyData[] = [
  {
    date: 'Mon',
    fullDate: '2025-01-13',
    coughCount: 8,
    attributePrevalence: {
      wet: 45.2,
      choking: 12.5,
      congestion: 28.7,
      stridor: 8.3,
      wheezing: 35.6,
    },
    coughsPerHour: 1.0,
    wheezeTimePercent: 5.3,
  },
  {
    date: 'Tue',
    fullDate: '2025-01-14',
    coughCount: 12,
    attributePrevalence: {
      wet: 58.3,
      choking: 18.7,
      congestion: 42.1,
      stridor: 12.4,
      wheezing: 48.9,
    },
    coughsPerHour: 1.5,
    wheezeTimePercent: 7.3,
  },
  {
    date: 'Wed',
    fullDate: '2025-01-15',
    coughCount: 5,
    attributePrevalence: {
      wet: 28.6,
      choking: 8.2,
      congestion: 18.4,
      stridor: 5.1,
      wheezing: 22.3,
    },
    coughsPerHour: 0.6,
    wheezeTimePercent: 3.3,
  },
  {
    date: 'Thu',
    fullDate: '2025-01-16',
    coughCount: 15,
    attributePrevalence: {
      wet: 72.4,
      choking: 24.8,
      congestion: 56.2,
      stridor: 18.6,
      wheezing: 62.1,
    },
    coughsPerHour: 1.9,
    wheezeTimePercent: 9.3,
  },
  {
    date: 'Fri',
    fullDate: '2025-01-17',
    coughCount: 3,
    attributePrevalence: {
      wet: 15.3,
      choking: 4.8,
      congestion: 12.1,
      stridor: 2.9,
      wheezing: 11.7,
    },
    coughsPerHour: 0.4,
    wheezeTimePercent: 1.8,
  },
  {
    date: 'Sat',
    fullDate: '2025-01-18',
    coughCount: 9,
    attributePrevalence: {
      wet: 52.1,
      choking: 14.3,
      congestion: 34.8,
      stridor: 10.2,
      wheezing: 41.5,
    },
    coughsPerHour: 1.1,
    wheezeTimePercent: 6.2,
  },
  {
    date: 'Sun',
    fullDate: '2025-01-19',
    coughCount: 7,
    attributePrevalence: {
      wet: 38.7,
      choking: 10.9,
      congestion: 25.3,
      stridor: 7.4,
      wheezing: 29.8,
    },
    coughsPerHour: 0.9,
    wheezeTimePercent: 4.5,
  },
];

