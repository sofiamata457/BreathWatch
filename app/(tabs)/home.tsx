import { CoughChart } from '@/components/CoughChart';
import { Paper, Typography, Box, CircularProgress } from '@mui/material';
import React, { useMemo, useEffect } from 'react';
import { useRecording } from '@/contexts/RecordingContext';
import { useRouter } from 'expo-router';
import { Colors } from '@/constants/theme';
import HistoryIcon from '@mui/icons-material/History';
import BarChartIcon from '@mui/icons-material/BarChart';
import SettingsIcon from '@mui/icons-material/Settings';
import HelpIcon from '@mui/icons-material/Help';
import InfoIcon from '@mui/icons-material/Info';

export default function HomePage() {
  const themeColors = Colors.dark;
  const router = useRouter();
  const { chunkResponses, finalSummary, fetchFinalSummary, isFetchingSummary, summaryProgress } = useRecording();

  // Default sample data - all zeros initially
  const defaultData = {
    counts: [0],
    labels: ['No Data'],
    breakdown: [
      { wet: 0, choking: 0, congestion: 0, stridor: 0, wheezing: 0 },
    ],
  };

  // Calculate data from ACTUAL recording data - NO FAKE DISTRIBUTIONS
  const chartData = useMemo(() => {
    // If we have a final summary, use ACTUAL data
    if (finalSummary) {
      const hourlyData = finalSummary.hourly_breakdown || [];
      const allEvents = finalSummary.cough_events || [];
      
      // Use hourly breakdown if available, otherwise show single total
      if (hourlyData.length > 0) {
        const labels = hourlyData.map((hour, idx) => `Hour ${idx + 1}`);
        const counts = hourlyData.map(hour => hour.cough_count); // KEEP AS COUNTS
        
        // Use PERCENTAGES from attribute_prevalence (from model output)
        const attr = finalSummary.attribute_prevalence || {};
        console.log('Attribute prevalence (hourly):', attr);
        const breakdown = [{
          wet: attr.wet || 0,                                    // PERCENTAGE
          choking: attr.choking || 0,                            // PERCENTAGE
          congestion: attr.congestion || 0,                      // PERCENTAGE
          stridor: attr.stridor || 0,                            // PERCENTAGE
          wheezing: attr.selfreported_wheezing || 0,             // PERCENTAGE
        }];
        console.log('Breakdown (hourly):', breakdown);
        
        return { counts, labels, breakdown };
      } else {
        // No hourly data - show single total
        const totalCoughs = allEvents.length;
        const labels = ['Recording'];
        const counts = [totalCoughs]; // KEEP AS COUNTS
        
        // Use PERCENTAGES from attribute_prevalence (from model output)
        const attr = finalSummary.attribute_prevalence || {};
        console.log('Attribute prevalence (single):', attr);
        const breakdown = [{
          wet: attr.wet || 0,                                    // PERCENTAGE
          choking: attr.choking || 0,                            // PERCENTAGE
          congestion: attr.congestion || 0,                      // PERCENTAGE
          stridor: attr.stridor || 0,                            // PERCENTAGE
          wheezing: attr.selfreported_wheezing || 0,             // PERCENTAGE
        }];
        console.log('Breakdown (single):', breakdown);
        
        return { counts, labels, breakdown };
      }
    }

    // If we have chunk responses, use ACTUAL data from chunks
    if (chunkResponses.length > 0) {
      const labels = chunkResponses.map((chunk, idx) => `Chunk ${chunk.chunk_index + 1}`);
      const counts = chunkResponses.map(chunk => chunk.cough_count);
      
      // Get ALL events from all chunks
      const allEvents = chunkResponses.flatMap(chunk => chunk.detected_events || []);
      
      // Calculate ACTUAL attribute counts from events
      const wetCount = allEvents.filter(e => e.tags?.includes('WET')).length;
      const chokingCount = allEvents.filter(e => e.tags?.includes('CHOKING')).length;
      const congestionCount = allEvents.filter(e => e.tags?.includes('CONGESTION')).length;
      const stridorCount = allEvents.filter(e => e.tags?.includes('STRIDOR')).length;
      const wheezingCount = allEvents.filter(e => e.tags?.includes('SELFREPORTED_WHEEZING')).length;
      
      // Single breakdown showing total attribute counts across all chunks
      const breakdown = [{
        wet: wetCount,
        choking: chokingCount,
        congestion: congestionCount,
        stridor: stridorCount,
        wheezing: wheezingCount,
      }];
      
      return { counts, labels, breakdown };
    }

    // Default: show zeros
    return defaultData;
  }, [chunkResponses, finalSummary]);

  // Show real-time updates when chunks come in
  useEffect(() => {
    if (chunkResponses.length > 0) {
      const lastChunk = chunkResponses[chunkResponses.length - 1];
      console.log('New chunk processed:', {
        chunkIndex: lastChunk.chunk_index,
        coughCount: lastChunk.cough_count,
        wheezeWindows: lastChunk.wheeze_windows,
      });
    }
  }, [chunkResponses]);

  const navigationCards = [
    {
      title: 'History',
      description: 'Past processed logs',
      icon: <HistoryIcon />,
      route: '/history',
      color: themeColors.bright,
    },
    {
      title: 'Statistics',
      description: 'Track trends',
      icon: <BarChartIcon />,
      route: '/statistics',
      color: themeColors.bright,
    },
    {
      title: 'Settings',
      description: 'Preferences',
      icon: <SettingsIcon />,
      route: '/settings',
      color: themeColors.bright,
    },
    {
      title: 'Help',
      description: 'FAQ & docs',
      icon: <HelpIcon />,
      route: '/help',
      color: themeColors.bright,
    },
    {
      title: 'About',
      description: 'Learn more',
      icon: <InfoIcon />,
      route: '/about',
      color: themeColors.bright,
    },
  ];

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `linear-gradient(135deg, ${themeColors.background} 0%, ${themeColors.backgroundGradient} 100%)`,
        display: 'flex',
        flexDirection: 'column',
        fontFamily: Colors.typography.fontFamily,
      }}
    >
      {/* Chart Section - Now at top */}
      <Box sx={{ flex: 1, overflowY: 'auto', pb: '80px' }}>
        <Paper sx={{ minHeight: '100%', background: 'transparent' }}>
         {isFetchingSummary && (
           <Box
             sx={{
               p: 3,
               textAlign: 'center',
               background: `linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 44, 191, 0.1) 100%)`,
               borderRadius: '12px',
               mx: 2,
               mb: 2,
               border: `1px solid rgba(0, 212, 255, 0.3)`,
             }}
           >
             <CircularProgress size={24} sx={{ mb: 2, color: themeColors.bright }} />
             <Typography variant="body1" sx={{ color: themeColors.text, fontWeight: 600, mb: 1 }}>
               {summaryProgress || 'Generating final summary...'}
             </Typography>
             <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7 }}>
               This may take up to 60 seconds depending on data size
             </Typography>
           </Box>
         )}
      {chunkResponses.length > 0 && !finalSummary && !isFetchingSummary && (
        <Box
          sx={{
            p: 2,
            textAlign: 'center',
            background: `linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 44, 191, 0.1) 100%)`,
            borderRadius: '12px',
            mx: 2,
            mb: 2,
            border: `1px solid rgba(0, 212, 255, 0.2)`,
          }}
        >
          <Typography variant="body2" sx={{ color: themeColors.text, fontWeight: 500 }}>
            {chunkResponses.length} chunk{chunkResponses.length !== 1 ? 's' : ''} processed •{' '}
            {chunkResponses.reduce((sum, chunk) => sum + chunk.cough_count, 0)} total coughs detected
          </Typography>
          <Typography variant="caption" sx={{ color: themeColors.text, opacity: 0.7, mt: 1, display: 'block' }}>
            Chart shows real-time data from processed chunks
          </Typography>
        </Box>
      )}
      {finalSummary && (
        <Box
          sx={{
            p: 2,
            textAlign: 'center',
            background: `linear-gradient(135deg, rgba(6, 255, 165, 0.15) 0%, rgba(0, 212, 255, 0.1) 100%)`,
            borderRadius: '12px',
            mx: 2,
            mb: 2,
            border: `1px solid ${themeColors.success}`,
          }}
        >
          <Typography variant="body2" sx={{ color: themeColors.success, fontWeight: 600 }}>
            Analysis Complete!
          </Typography>
          <Typography variant="body2" sx={{ color: themeColors.text, mt: 0.5, opacity: 0.9 }}>
            {finalSummary.coughs_per_hour.toFixed(1)} coughs/hour •{' '}
            {finalSummary.wheeze_time_percent.toFixed(1)}% wheeze time
          </Typography>
        </Box>
      )}
      <CoughChart
        counts={chartData.counts}
        labels={chartData.labels}
        breakdown={chartData.breakdown}
      />
        </Paper>
      </Box>

      {/* Action Bar - Now at bottom */}
      <Box
        sx={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          background: `linear-gradient(to top, ${themeColors.background} 0%, ${themeColors.backgroundGradient} 100%)`,
          borderTop: `1px solid ${themeColors.secondary}`,
          backdropFilter: 'blur(10px)',
          zIndex: 1000,
          boxShadow: `0 -4px 20px rgba(0, 0, 0, 0.3)`,
        }}
      >
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-around',
            alignItems: 'center',
            px: 1,
            py: 1.5,
            maxWidth: '100%',
            overflowX: 'auto',
            '&::-webkit-scrollbar': {
              display: 'none',
            },
            scrollbarWidth: 'none',
          }}
        >
          {navigationCards.map((card, index) => (
            <Box
              key={index}
              onClick={() => router.push(card.route)}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minWidth: { xs: '60px', sm: '80px' },
                px: { xs: 1, sm: 2 },
                py: 1,
                cursor: 'pointer',
                borderRadius: '12px',
                transition: 'all 0.2s ease',
                position: 'relative',
                '&:hover': {
                  background: `linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(123, 44, 191, 0.15) 100%)`,
                  transform: 'translateY(-2px)',
                },
                '&:active': {
                  transform: 'translateY(0)',
                },
              }}
            >
              <Box
                sx={{
                  color: card.color,
                  fontSize: { xs: '24px', sm: '28px' },
                  mb: 0.5,
                  filter: 'drop-shadow(0 2px 4px rgba(0, 212, 255, 0.3))',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {card.icon}
              </Box>
              <Typography
                variant="caption"
                sx={{
                  color: themeColors.text,
                  fontWeight: 500,
                  fontSize: { xs: '0.65rem', sm: '0.75rem' },
                  textAlign: 'center',
                  whiteSpace: 'nowrap',
                  opacity: 0.9,
                }}
              >
                {card.title}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
}
