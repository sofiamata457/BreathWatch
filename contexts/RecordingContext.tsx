/**
 * Recording context for sharing recording state across the app
 */

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { useAudioRecorder, ChunkProcessResponse } from '@/hooks/useAudioRecorder';
import { getFinalSummary, NightlySummary } from '@/services/api';
import { saveRecordingToHistory, getSettings } from '@/services/storage';

interface RecordingContextType {
  recordingState: ReturnType<typeof useAudioRecorder>['recordingState'];
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<void>;
  finalSummary: NightlySummary | null;
  isFetchingSummary: boolean;
  summaryProgress: string | null;
  fetchFinalSummary: (symptomForm?: any) => Promise<void>;
  chunkResponses: ChunkProcessResponse[];
}

const RecordingContext = createContext<RecordingContextType | undefined>(undefined);

export function RecordingProvider({ children }: { children: ReactNode }) {
  const [finalSummary, setFinalSummary] = useState<NightlySummary | null>(null);
  const [isFetchingSummary, setIsFetchingSummary] = useState(false);
  const [summaryProgress, setSummaryProgress] = useState<string | null>(null);
  const [chunkResponses, setChunkResponses] = useState<ChunkProcessResponse[]>([]);

  const handleChunkProcessed = useCallback((response: ChunkProcessResponse) => {
    setChunkResponses((prev) => [...prev, response]);
  }, []);

  const handleError = useCallback((error: string) => {
    console.error('Recording error:', error);
    // Could show a toast notification here
  }, []);

  const { recordingState, startRecording: startRec, stopRecording: stopRec, sendFinalChunk } =
    useAudioRecorder(handleChunkProcessed, handleError);

  const fetchFinalSummary = useCallback(
    async (symptomForm?: any) => {
      if (!recordingState.sessionId) {
        return;
      }

      setIsFetchingSummary(true);
      setSummaryProgress('Initializing...');
      
      try {
        // Update progress messages
        const progressMessages: string[] = [];
        const updateProgress = (message: string) => {
          progressMessages.push(message);
          setSummaryProgress(message);
          console.log(`📊 Summary progress: ${message}`);
        };

        const summary = await getFinalSummary(recordingState.sessionId, symptomForm, updateProgress);
        
        setFinalSummary(summary);
        setSummaryProgress('Complete!');
        
        // Auto-save to history if enabled
        const settings = await getSettings();
        if (settings.autoSaveRecordings) {
          updateProgress('Saving to history...');
          await saveRecordingToHistory(summary);
        }
        
        // Clear progress after a moment
        setTimeout(() => {
          setSummaryProgress(null);
        }, 2000);
      } catch (error) {
        console.error('Failed to fetch final summary:', error);
        const errorMessage = error instanceof Error ? error.message : 'Failed to fetch summary';
        setSummaryProgress(`Error: ${errorMessage}`);
        handleError(errorMessage);
        
        // Clear progress after showing error
        setTimeout(() => {
          setSummaryProgress(null);
        }, 5000);
      } finally {
        setIsFetchingSummary(false);
      }
    },
    [recordingState.sessionId, handleError]
  );

  const startRecording = useCallback(async () => {
    setChunkResponses([]);
    setFinalSummary(null);
    await startRec();
  }, [startRec]);

  const stopRecording = useCallback(async () => {
    const wasRecording = recordingState.isRecording;
    const sessionId = recordingState.sessionId;
    
    // First, stop the recording (this will trigger final data events)
    await stopRec();
    
    // Then send final chunk (even if recording was very short)
    // This ensures any remaining audio data is sent as a chunk
    if (wasRecording && sessionId) {
      try {
        console.log('📤 Sending final chunk (may be short recording)...');
        await sendFinalChunk();
        console.log('✅ Final chunk sent successfully');
        // Wait a moment for the chunk to be processed and added to chunkResponses
        await new Promise((resolve) => setTimeout(resolve, 1000));
      } catch (error) {
        console.error('❌ Error sending final chunk:', error);
        // Continue even if final chunk fails - we'll still try to get summary
      }
    }
    
    // Automatically fetch final summary if we have a session
    // Check chunkResponses again after final chunk might have been added
    if (wasRecording && sessionId) {
      // Wait a bit longer to ensure backend has processed everything
      setTimeout(async () => {
        try {
          console.log('📊 Fetching final summary for session:', sessionId);
          // Get updated chunk count
          const updatedChunkCount = chunkResponses.length;
          console.log('📊 Current chunk count:', updatedChunkCount);
          
          if (updatedChunkCount === 0) {
            console.warn('⚠️ No chunks were processed. This may indicate the recording was too short or chunks failed to send.');
            handleError('No audio chunks were processed. Please ensure you recorded for at least a few seconds.');
            return;
          }
          
          await fetchFinalSummary();
          console.log('✅ Final summary fetched successfully');
        } catch (error) {
          console.error('❌ Error fetching final summary:', error);
          // Show error to user
          const errorMsg = error instanceof Error ? error.message : 'Unknown error';
          if (errorMsg.includes('not found')) {
            handleError('Session not found. No chunks were successfully processed. Please try recording again.');
          } else {
            handleError(`Failed to fetch summary: ${errorMsg}`);
          }
        }
      }, 2000); // Wait 2 seconds for backend to process
    }
  }, [stopRec, sendFinalChunk, recordingState.isRecording, recordingState.sessionId, chunkResponses.length, fetchFinalSummary, handleError]);

  return (
    <RecordingContext.Provider
      value={{
        recordingState,
        startRecording,
        stopRecording,
        finalSummary,
        isFetchingSummary,
        summaryProgress,
        fetchFinalSummary,
        chunkResponses,
      }}
    >
      {children}
    </RecordingContext.Provider>
  );
}

export function useRecording() {
  const context = useContext(RecordingContext);
  if (context === undefined) {
    throw new Error('useRecording must be used within a RecordingProvider');
  }
  return context;
}

