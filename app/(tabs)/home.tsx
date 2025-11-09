import { CoughChart } from '@/components/CoughChart';
import { Paper, Typography, Box } from '@mui/material';
import React, { useMemo, useEffect } from 'react';
import { useRecording } from '@/contexts/RecordingContext';

export default function HomePage() {
  const { chunkResponses, finalSummary, fetchFinalSummary, isFetchingSummary } = useRecording();

  // Default sample data (original styling) - all zeros initially
  const defaultData = {
    counts: [0, 0, 0, 0, 0, 0, 0],
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    breakdown: [
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
      { wet: 0, dry: 0 },
    ],
  };

  // Calculate data from chunk responses or final summary
  const chartData = useMemo(() => {
    // If we have a final summary, use that (most complete data)
    if (finalSummary) {
      // Use hourly breakdown for the chart
      const hourlyData = finalSummary.hourly_breakdown || [];
      
      // Calculate counts and labels from hourly data
      const counts = hourlyData.map((hour) => hour.cough_count);
      const labels = hourlyData.map((hour) => `Hour ${hour.hour + 1}`);
      
      // Calculate breakdown (wet vs dry) from attribute prevalence
      // This is a simplified calculation - in reality, you'd want to track this per event
      const breakdown = hourlyData.map((hour) => {
        // Estimate wet/dry based on attribute prevalence in the summary
        // This is approximate - ideally you'd track this per hour
        const totalCoughs = hour.cough_count;
        const wetEstimate = Math.round(
          (totalCoughs * finalSummary.attribute_prevalence.wet) / 100
        );
        const dryEstimate = totalCoughs - wetEstimate;
        return { wet: wetEstimate, dry: dryEstimate };
      });

      return { counts, labels, breakdown };
    }

    // Otherwise, use chunk responses to build up data
    if (chunkResponses.length > 0) {
      // For short recordings, show data by chunk instead of by hour
      // Each chunk represents 10 minutes of data
      const totalCoughs = chunkResponses.reduce((sum, chunk) => sum + chunk.cough_count, 0);
      
      // If we have multiple chunks, group by hour
      // If we have just 1-2 chunks, show them individually
      if (chunkResponses.length <= 2) {
        // Show individual chunks
        const counts = chunkResponses.map((chunk) => chunk.cough_count);
        const labels = chunkResponses.map((chunk, idx) => `Chunk ${idx + 1}`);
        
        // Calculate breakdown from events if available
        const breakdown = chunkResponses.map((chunk) => {
          // Try to estimate wet/dry from detected events
          // This is a simplified approach
          const totalEvents = chunk.detected_events?.length || 0;
          // For now, split 50/50 as placeholder
          return {
            wet: Math.floor(totalEvents * 0.5),
            dry: Math.ceil(totalEvents * 0.5),
          };
        });

        return { counts, labels, breakdown };
      } else {
        // Group by hour (each chunk is 10 minutes = 1/6 of an hour)
        const hourlyMap = new Map<number, { coughs: number; events: any[] }>();
        
        chunkResponses.forEach((response, chunkIndex) => {
          const hourIndex = Math.floor(chunkIndex / 6); // 6 chunks per hour (10 min each)
          
          if (!hourlyMap.has(hourIndex)) {
            hourlyMap.set(hourIndex, { coughs: 0, events: [] });
          }
          
          const hourData = hourlyMap.get(hourIndex)!;
          hourData.coughs += response.cough_count;
          hourData.events.push(...(response.detected_events || []));
        });

        // Convert to arrays
        const hours = Array.from(hourlyMap.keys()).sort((a, b) => a - b);
        const counts = hours.map((hour) => hourlyMap.get(hour)!.coughs);
        const labels = hours.map((hour) => `Hour ${hour + 1}`);
        
        // Calculate breakdown from events
        const breakdown = hours.map((hour) => {
          const events = hourlyMap.get(hour)!.events;
          const totalEvents = events.length;
          return {
            wet: Math.floor(totalEvents * 0.5),
            dry: Math.ceil(totalEvents * 0.5),
          };
        });

        return { counts, labels, breakdown };
      }
    }

    // Default: show original styled sample data
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

  return (
    <Paper sx={{ minHeight: '100%' }}>
      {isFetchingSummary && (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            Generating final summary...
          </Typography>
        </Box>
      )}
      {chunkResponses.length > 0 && !finalSummary && !isFetchingSummary && (
        <Box sx={{ p: 2, textAlign: 'center', backgroundColor: 'rgba(0,0,0,0.05)' }}>
          <Typography variant="body2" color="text.secondary">
            {chunkResponses.length} chunk{chunkResponses.length !== 1 ? 's' : ''} processed •{' '}
            {chunkResponses.reduce((sum, chunk) => sum + chunk.cough_count, 0)} total coughs
            detected
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Chart shows real-time data from processed chunks
          </Typography>
        </Box>
      )}
      {finalSummary && (
        <Box sx={{ p: 2, textAlign: 'center', backgroundColor: 'rgba(76, 175, 80, 0.1)' }}>
          <Typography variant="body2" color="success.main" sx={{ fontWeight: 600 }}>
            Analysis Complete!
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
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
  );
}
