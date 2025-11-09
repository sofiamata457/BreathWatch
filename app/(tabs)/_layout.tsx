import { Colors } from '@/constants/theme';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';
import { Box, Fab, CircularProgress, Snackbar, Alert, Chip, Typography } from '@mui/material';
import { Slot, usePathname, useRouter, useSegments } from 'expo-router';
import React, { useState, useEffect } from 'react';
import { useRecording } from '@/contexts/RecordingContext';

// Define enum for tab names
enum TabName {
  Home = 'home',
  Record = 'record',
  About = 'about',
}

export default function TabLayout() {
  const themeColors = Colors.dark;
  const { recordingState, startRecording, stopRecording, chunkResponses, finalSummary, isFetchingSummary, summaryProgress } = useRecording();
  const [errorSnackbar, setErrorSnackbar] = useState<string | null>(null);
  const [successSnackbar, setSuccessSnackbar] = useState<string | null>(null);
  const [recordingDuration, setRecordingDuration] = useState<number>(0);

  const router = useRouter();
  const segments = useSegments(); // current route segments
  const pathname = usePathname(); // e.g., '/', '/about', '/record'

  const currentTab: TabName = (() => {
    switch (pathname) {
      case '/about':
        return TabName.About;
      case '/record':
        return TabName.Record;
      case '/':
        return TabName.Home;
      default:
        return TabName.Home;
    }
  })();

  const handleChange = (_event: React.SyntheticEvent, newValue: TabName) => {
    router.replace(`/${newValue === TabName.Home ? '' : newValue}`);
    console.log(currentTab);
  };

  const handleRecordClick = async () => {
    try {
      if (recordingState.isRecording) {
        setSuccessSnackbar('Stopping recording and processing final chunk...');
        await stopRecording();
        setSuccessSnackbar('Recording stopped. Processing data...');
        setTimeout(() => setSuccessSnackbar(null), 2000);
      } else {
        setSuccessSnackbar('Starting recording...');
        await startRecording();
        setSuccessSnackbar('Recording started!');
        setTimeout(() => setSuccessSnackbar(null), 2000);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to toggle recording';
      setErrorSnackbar(errorMessage);
      setSuccessSnackbar(null);
    }
  };

  // Show error from recording state
  React.useEffect(() => {
    if (recordingState.error) {
      setErrorSnackbar(recordingState.error);
    }
  }, [recordingState.error]);

  // Show success when chunks are processed
  useEffect(() => {
    if (chunkResponses.length > 0) {
      const lastChunk = chunkResponses[chunkResponses.length - 1];
      setSuccessSnackbar(
        `Chunk processed: ${lastChunk.cough_count} coughs detected, ${lastChunk.wheeze_windows} wheeze windows`
      );
      // Auto-hide after 3 seconds
      setTimeout(() => setSuccessSnackbar(null), 3000);
    }
  }, [chunkResponses.length]);

  // Show success when final summary is ready
  useEffect(() => {
    if (finalSummary) {
      setSuccessSnackbar(
        `Analysis complete! ${finalSummary.coughs_per_hour.toFixed(1)} coughs/hour detected`
      );
      setTimeout(() => setSuccessSnackbar(null), 5000);
    }
  }, [finalSummary]);

  // Track recording duration
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (recordingState.isRecording) {
      interval = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } else {
      setRecordingDuration(0);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [recordingState.isRecording]);

  // Format duration as MM:SS
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#DDD6F3',
        overflow: 'hidden',
      }}
    >
      {/* Processing Summary Banner */}
      {isFetchingSummary && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            backgroundColor: '#2196F3',
            color: 'white',
            padding: '12px 20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 2,
            zIndex: 1001,
            boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          }}
        >
          <CircularProgress size={16} sx={{ color: 'white' }} />
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            {summaryProgress || 'Processing final summary...'}
          </Typography>
        </Box>
      )}

      {/* Recording Indicator Banner */}
      {recordingState.isRecording && !isFetchingSummary && (
        <Box
          sx={{
            position: 'fixed',
            top: isFetchingSummary ? '56px' : 0,
            left: 0,
            right: 0,
            backgroundColor: '#ff4444',
            color: 'white',
            padding: '12px 20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 2,
            zIndex: 1000,
            boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          }}
        >
          <FiberManualRecordIcon sx={{ fontSize: 16, animation: 'pulse 1.5s infinite' }} />
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Recording... {formatDuration(recordingDuration)}
          </Typography>
          {recordingState.isProcessing && (
            <CircularProgress size={16} sx={{ color: 'white' }} />
          )}
          {chunkResponses.length > 0 && (
            <Chip
              label={`${chunkResponses.length} chunk${chunkResponses.length !== 1 ? 's' : ''} processed`}
              size="small"
              sx={{
                backgroundColor: 'rgba(255,255,255,0.2)',
                color: 'white',
                height: '24px',
              }}
            />
          )}
          <style>
            {`
              @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
              }
            `}
          </style>
        </Box>
      )}

      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
          paddingTop: (recordingState.isRecording || isFetchingSummary) ? '56px' : 0,
        }}
      >
        <Slot />
      </Box>

      {/* Floating Record Button */}
      <Fab
        onClick={handleRecordClick}
        disabled={recordingState.isProcessing}
        style={{
          position: 'fixed',
          bottom: 120,
          right: 40,
          zIndex: 1002,
          backgroundColor: recordingState.isRecording ? '#ff4444' : themeColors.text,
          color: themeColors.background,
        }}
      >
        {recordingState.isProcessing ? (
          <CircularProgress size={24} color="inherit" />
        ) : recordingState.isRecording ? (
          <StopIcon />
        ) : (
          <MicIcon />
        )}
      </Fab>

      {/* Error Snackbar */}
      <Snackbar
        open={!!errorSnackbar}
        autoHideDuration={6000}
        onClose={() => setErrorSnackbar(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setErrorSnackbar(null)} severity="error" sx={{ width: '100%' }}>
          {errorSnackbar}
        </Alert>
      </Snackbar>

      {/* Success Snackbar */}
      <Snackbar
        open={!!successSnackbar}
        autoHideDuration={3000}
        onClose={() => setSuccessSnackbar(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={() => setSuccessSnackbar(null)} severity="success" sx={{ width: '100%' }}>
          {successSnackbar}
        </Alert>
      </Snackbar>

      {/* Bottom Navigation
      <BottomNavigation
        value={currentTab}
        onChange={handleChange}
        sx={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          color: themeColors.text,
        }}
      >
        <BottomNavigationAction
          // label="Home"
          value={TabName.Home}
          icon={<HomeIcon />}
          sx={{ color: currentTab === TabName.Home ? themeColors.tint : 'gray' }}
          onClick={() => router.replace('/')}
        />
        <BottomNavigationAction
          // label="About"
          value={TabName.About}
          icon={<InfoIcon />}
          sx={{ color: currentTab === TabName.About ? themeColors.tint : 'gray' }}
          onClick={() => router.replace('/about')}
        />
      </BottomNavigation> */}
    </Box>
  );
}
