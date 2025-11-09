import React, { useState, useMemo } from 'react';
import { Box, Typography, Chip } from '@mui/material';
import { BarChart } from '@mui/x-charts/BarChart';
import { Colors } from '@/constants/theme';
import { DailyData } from '@/utils/mockWeeklyData';

interface WeeklyCoughChartProps {
  weeklyData: DailyData[];
}

export const WeeklyCoughChart: React.FC<WeeklyCoughChartProps> = ({ weeklyData }) => {
  const themeColors = Colors.dark;
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  const { counts, labels, breakdown } = useMemo(() => {
    return {
      counts: weeklyData.map((day) => day.coughCount),
      labels: weeklyData.map((day) => day.date),
      breakdown: weeklyData.map((day) => day.attributePrevalence),
    };
  }, [weeklyData]);

  const handleItemClick = (event: any, itemIdentifier: any) => {
    if (itemIdentifier && typeof itemIdentifier.dataIndex === 'number') {
      const newIndex = itemIdentifier.dataIndex;
      setSelectedIndex(newIndex === selectedIndex ? null : newIndex);
    }
  };

  const selectedBreakdown = selectedIndex !== null ? breakdown[selectedIndex] : null;

  return (
    <Box
      sx={{
        width: '100%',
        maxWidth: 'calc(100dvw - 40px)',
        marginLeft: '20px',
        my: '30px',
        background: `linear-gradient(-45deg, ${themeColors.secondary} 25%, ${themeColors.tertiary})`,
        borderRadius: '25px',
        boxShadow: `3px 3px 0 ${themeColors.text}`,
        padding: 3,
      }}
    >
      <Typography variant="h6" sx={{ color: themeColors.text, fontWeight: 700, mb: 1 }}>
        Weekly Cough Overview
      </Typography>
      <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.75, mb: 2 }}>
        {selectedIndex !== null
          ? `Selected: ${labels[selectedIndex]} - ${weeklyData[selectedIndex].coughCount} coughs`
          : 'Click on a bar to see attribute breakdown'}
      </Typography>

      {/* Main bar chart - cough counts */}
      <BarChart
        height={300}
        series={[
          {
            data: counts,
            label: 'Cough Count',
            color: themeColors.bright,
            id: 'cough-count',
          },
        ]}
        xAxis={[
          {
            data: labels,
            scaleType: 'band',
            tickLabelStyle: { fill: themeColors.text },
            labelStyle: { fill: themeColors.text },
            sx: {
              '& .MuiChartsAxis-line': {
                stroke: themeColors.text,
              },
              '& .MuiChartsAxis-tick': {
                stroke: themeColors.text,
              },
            },
          },
        ]}
        yAxis={[
          {
            tickLabelStyle: { fill: themeColors.text },
            labelStyle: { fill: themeColors.text },
            sx: {
              '& .MuiChartsAxis-line': {
                stroke: themeColors.text,
              },
              '& .MuiChartsAxis-tick': {
                stroke: themeColors.text,
              },
            },
          },
        ]}
        onItemClick={handleItemClick}
        slotProps={{
          bar: {
            rx: 8,
            style: {
              cursor: 'pointer',
              transition: 'opacity 0.2s',
            },
          },
        }}
        sx={{
          '& .MuiBarElement-root': {
            fill: themeColors.bright,
            '&:hover': {
              opacity: 0.8,
            },
          },
        }}
      />

      {/* Breakdown chart - attribute percentages */}
      {selectedBreakdown && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="subtitle1" sx={{ color: themeColors.text, fontWeight: 600, mb: 2 }}>
            Attribute Prevalence (%) - {labels[selectedIndex]}
          </Typography>
          <BarChart
            height={200}
            layout="horizontal"
            xAxis={[
              {
                min: 0,
                max: 100,
                tickLabelStyle: { fill: themeColors.text },
                labelStyle: { fill: themeColors.text },
                sx: {
                  '& .MuiChartsAxis-line': {
                    stroke: themeColors.text,
                  },
                  '& .MuiChartsAxis-tick': {
                    stroke: themeColors.text,
                  },
                },
              },
            ]}
            yAxis={[
              {
                data: ['Wet', 'Choking', 'Congestion', 'Stridor', 'Wheezing'],
                scaleType: 'band',
                tickLabelStyle: { fill: themeColors.text },
                labelStyle: { fill: themeColors.text },
                sx: {
                  '& .MuiChartsAxis-line': {
                    stroke: themeColors.text,
                  },
                  '& .MuiChartsAxis-tick': {
                    stroke: themeColors.text,
                  },
                },
              },
            ]}
            series={[
              {
                data: [
                  selectedBreakdown.wet,
                  selectedBreakdown.choking,
                  selectedBreakdown.congestion,
                  selectedBreakdown.stridor,
                  selectedBreakdown.wheezing,
                ],
                label: 'Attribute Prevalence',
                color: themeColors.bright,
              },
            ]}
            spacing={0.3}
            grid={{ vertical: true, horizontal: true }}
            slotProps={{
              tooltip: { trigger: 'none' },
            }}
          />
        </Box>
      )}

      {selectedIndex === null && (
        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Typography variant="body2" sx={{ color: themeColors.text, opacity: 0.7 }}>
            Click on a day's bar above to see attribute breakdown
          </Typography>
        </Box>
      )}
    </Box>
  );
};

