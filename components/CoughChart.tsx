import { Colors } from '@/constants/theme';
import { Box, Typography } from '@mui/material';
import { BarItemIdentifier } from '@mui/x-charts';
import { BarChart } from '@mui/x-charts/BarChart';
import { labelMarkClasses } from '@mui/x-charts/ChartsLabel';
import React from 'react';

interface CoughChartProps {
  counts: number[];
  labels: string[];
  breakdown?: { wet: number; dry: number }[]; // optional distribution per day
}

export const CoughChart: React.FC<CoughChartProps> = ({ counts, labels, breakdown }) => {
  const themeColors = Colors.dark;

  // Track selected day
  const [selectedIndex, setSelectedIndex] = React.useState<number | null>(null);

  const handleItemClick = (
    _event: React.MouseEvent<SVGElement, MouseEvent>,
    barItemIdentifier: BarItemIdentifier,
  ) => {
    console.log(1);
    setSelectedIndex(barItemIdentifier.dataIndex);
  };

  return (
    <Box
      sx={{
        width: '100%',
        minHeight: '100dvh',
        mx: 'auto',
        my: 0,
        pt: '50px',
        backgroundColor: themeColors.background,
        color: themeColors.text,
        fontFamily: Colors.typography.fontFamily,
      }}
    >
      <Typography variant="h6" gutterBottom align="center" sx={{ color: themeColors.text }}>
        Nightly Cough Count
      </Typography>

      {/* Total cough count chart */}
      <Box
        sx={{
          maxWidth: 'calc(100dvw - 40px)',
          marginLeft: '20px',
          backgroundColor: themeColors.secondary,
          borderRadius: '25px',
          boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
          padding: 0,
        }}
      >
        <BarChart
          sx={{
            height: '250px',
            maxWidth: 'calc(100dvw - 40px)',
          }}
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
          series={[
            {
              data: counts,
              label: 'Coughs',
              color: themeColors.bright,
            },
          ]}
          spacing={0.3}
          borderRadius={4}
          onItemClick={handleItemClick}
          grid={{ vertical: true, horizontal: true }}
          slotProps={{
            tooltip: { trigger: 'none' },
            legend: {
              sx: {
                color: themeColors.text,
                [`.${labelMarkClasses.fill}`]: {
                  fill: themeColors.text,
                },
              },
            },
          }}
          layout="vertical"
        />

        {/* Breakdown chart */}
        <Box sx={{ mt: 6 }}>
          {selectedIndex === null ? (
            <Typography variant="body1" align="center" sx={{ py: 10, color: themeColors.text }}>
              Select a day
            </Typography>
          ) : breakdown && breakdown[selectedIndex] ? (
            <BarChart
              height={150}
              layout="horizontal"
              xAxis={[
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
              yAxis={[
                {
                  data: ['Wet', 'Dry'],
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
                  data: [breakdown[selectedIndex].wet, breakdown[selectedIndex].dry],
                  label: 'Cough Distribution',
                  color: themeColors.text,
                },
              ]}
              spacing={0.3}
              grid={{ vertical: true, horizontal: true }}
              slotProps={{
                tooltip: { trigger: 'none' },
                legend: {
                  sx: {
                    color: themeColors.text,
                    [`.${labelMarkClasses.fill}`]: {
                      fill: themeColors.text,
                    },
                  },
                },
              }}
            />
          ) : (
            <Typography variant="body1" align="center" sx={{ py: 10, color: themeColors.text }}>
              No data available
            </Typography>
          )}
        </Box>
      </Box>
    </Box>
  );
};
