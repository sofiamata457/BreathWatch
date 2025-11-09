/**
 * Audio recording hook for nightly recording with 10-minute chunking
 * Supports both web (MediaRecorder) and native (expo-av)
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { Audio } from 'expo-av';
import { processChunk, ChunkProcessResponse } from '@/services/api';
import { Platform } from 'react-native';

const CHUNK_DURATION_MS = 10 * 60 * 1000; // 10 minutes in milliseconds
const SAMPLE_RATE = 16000; // 16kHz sample rate (matches backend)

export interface RecordingState {
  isRecording: boolean;
  isProcessing: boolean;
  sessionId: string | null;
  chunkIndex: number;
  error: string | null;
  chunkResponses: ChunkProcessResponse[];
}

export interface UseAudioRecorderReturn {
  recordingState: RecordingState;
  startRecording: () => Promise<void>;
  stopRecording: () => Promise<void>;
  sendFinalChunk: () => Promise<void>;
}

/**
 * Convert audio blob to WAV format (if needed)
 * MediaRecorder on web already produces WAV/WebM, so we may just need to pass it through
 */
async function prepareAudioBlob(blob: Blob): Promise<Blob> {
  // For web, MediaRecorder typically produces WebM or WAV
  // Backend expects WAV, so we may need conversion
  // For MVP, we'll try sending as-is and see if backend accepts it
  // If not, we'd need a WebM to WAV converter library
  
  // Check if it's already WAV
  if (blob.type === 'audio/wav' || blob.type === 'audio/wave') {
    return blob;
  }
  
  // For now, return as-is - backend may accept WebM or we'll need to add conversion
  return blob;
}

export function useAudioRecorder(
  onChunkProcessed?: (response: ChunkProcessResponse) => void,
  onError?: (error: string) => void
): UseAudioRecorderReturn {
  const [recordingState, setRecordingState] = useState<RecordingState>({
    isRecording: false,
    isProcessing: false,
    sessionId: null,
    chunkIndex: 0,
    error: null,
    chunkResponses: [],
  });

  // For web: MediaRecorder
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const webChunksRef = useRef<Blob[]>([]); // Store chunks for web recording
  
  // For native: expo-av
  const recordingRef = useRef<Audio.Recording | null>(null);
  
  // Common refs
  const chunkIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const lastChunkTimeRef = useRef<number | null>(null);
  const isWeb = Platform.OS === 'web';

  // Generate unique session ID
  const generateSessionId = useCallback(() => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Process and send a chunk
  const processAndSendChunk = useCallback(
    async (audioBlob: Blob, chunkIndex: number, sessionId: string) => {
      try {
        setRecordingState((prev) => ({ ...prev, isProcessing: true, error: null }));

        // Prepare audio blob (convert to WAV if needed)
        const wavBlob = await prepareAudioBlob(audioBlob);

        // Send to backend
        const response = await processChunk(wavBlob, chunkIndex, sessionId);

        setRecordingState((prev) => ({
          ...prev,
          chunkResponses: [...prev.chunkResponses, response],
          chunkIndex: chunkIndex + 1,
          isProcessing: false,
        }));

        if (onChunkProcessed) {
          onChunkProcessed(response);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        setRecordingState((prev) => ({
          ...prev,
          error: errorMessage,
          isProcessing: false,
        }));
        if (onError) {
          onError(errorMessage);
        }
        throw error;
      }
    },
    [onChunkProcessed, onError]
  );

  // Start recording (Web implementation)
  const startRecordingWeb = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const sessionId = generateSessionId();
      let chunkIndex = 0;
      lastChunkTimeRef.current = Date.now();
      webChunksRef.current = []; // Reset chunks

      // Create MediaRecorder with timeslice for chunking
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus', // WebM is widely supported
      });

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          webChunksRef.current.push(event.data);
          
          // When we have a complete chunk (10 minutes), process it
          if (Date.now() - (lastChunkTimeRef.current || 0) >= CHUNK_DURATION_MS) {
            const chunkBlob = new Blob(webChunksRef.current, { type: 'audio/webm' });
            await processAndSendChunk(chunkBlob, chunkIndex, sessionId);
            webChunksRef.current = []; // Clear chunks
            chunkIndex++;
            lastChunkTimeRef.current = Date.now();
          }
        }
      };

      mediaRecorder.onerror = (event) => {
        const error = new Error('MediaRecorder error');
        if (onError) {
          onError(error.message);
        }
      };

      // Start recording with timeslice (10 minutes)
      mediaRecorder.start(CHUNK_DURATION_MS);
      mediaRecorderRef.current = mediaRecorder;

      setRecordingState({
        isRecording: true,
        isProcessing: false,
        sessionId,
        chunkIndex: 0,
        error: null,
        chunkResponses: [],
      });

      startTimeRef.current = Date.now();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to start recording';
      setRecordingState((prev) => ({
        ...prev,
        error: errorMessage,
        isRecording: false,
      }));
      if (onError) {
        onError(errorMessage);
      }
    }
  }, [generateSessionId, processAndSendChunk, onError]);

  // Start recording (Native implementation)
  const startRecordingNative = useCallback(async () => {
    try {
      // Request permissions
      const { status } = await Audio.requestPermissionsAsync();
      if (status !== 'granted') {
        throw new Error('Audio recording permission not granted');
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const sessionId = generateSessionId();

      // Create recording with proper settings
      const { recording } = await Audio.Recording.createAsync(
        {
          ...Audio.RecordingOptionsPresets.HIGH_QUALITY,
          sampleRate: SAMPLE_RATE,
          numberOfChannels: 1, // Mono
          bitRate: 128000,
        },
        undefined, // No status callback needed
        0 // No interval
      );

      recordingRef.current = recording;
      startTimeRef.current = Date.now();
      lastChunkTimeRef.current = Date.now();

      setRecordingState({
        isRecording: true,
        isProcessing: false,
        sessionId,
        chunkIndex: 0,
        error: null,
        chunkResponses: [],
      });

      // Set up interval to extract and send chunks
      chunkIntervalRef.current = setInterval(async () => {
        if (!recordingRef.current || !recordingState.sessionId) return;

        try {
          const currentState = recordingState;
          if (!currentState.isRecording) return;

          // Stop current recording to get the file
          const uri = recordingRef.current.getURI();
          if (!uri) {
            console.warn('No recording URI available yet');
            return;
          }

          // Get the recording file
          const { sound } = await Audio.Sound.createAsync({ uri }, { shouldPlay: false });
          
          // For native, we need to read the file and send it
          // This is a simplified approach - in production, you'd want to:
          // 1. Keep recording continuously
          // 2. Extract the last 10 minutes from the buffer
          // 3. Send that chunk
          // 4. Continue recording

          // For MVP, we'll stop, send, and restart
          await recordingRef.current.stopAndUnloadAsync();
          
          // Fetch the file as blob
          const response = await fetch(uri);
          const blob = await response.blob();

          // Send chunk
          await processAndSendChunk(blob, currentState.chunkIndex, currentState.sessionId);

          // Start new recording for next chunk
          const { recording: newRecording } = await Audio.Recording.createAsync(
            {
              ...Audio.RecordingOptionsPresets.HIGH_QUALITY,
              sampleRate: SAMPLE_RATE,
              numberOfChannels: 1,
              bitRate: 128000,
            }
          );
          recordingRef.current = newRecording;
          lastChunkTimeRef.current = Date.now();
        } catch (error) {
          console.error('Error processing chunk:', error);
          if (onError) {
            onError(`Failed to process chunk: ${error}`);
          }
        }
      }, CHUNK_DURATION_MS);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to start recording';
      setRecordingState((prev) => ({
        ...prev,
        error: errorMessage,
        isRecording: false,
      }));
      if (onError) {
        onError(errorMessage);
      }
    }
  }, [generateSessionId, processAndSendChunk, onError, recordingState]);

  // Start recording (unified)
  const startRecording = useCallback(async () => {
    if (isWeb) {
      await startRecordingWeb();
    } else {
      await startRecordingNative();
    }
  }, [isWeb, startRecordingWeb, startRecordingNative]);

  // Stop recording (Web)
  const stopRecordingWeb = useCallback(async () => {
    return new Promise<void>((resolve) => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        // Request any remaining data before stopping
        mediaRecorderRef.current.requestData();
        mediaRecorderRef.current.stop();
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
      mediaRecorderRef.current = null;
      resolve();
    });
  }, []);

  // Stop recording (Native)
  const stopRecordingNative = useCallback(async () => {
    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
      chunkIntervalRef.current = null;
    }
    if (recordingRef.current) {
      await recordingRef.current.stopAndUnloadAsync();
      recordingRef.current = null;
    }
  }, []);

  // Stop recording (unified)
  const stopRecording = useCallback(async () => {
    try {
      if (isWeb) {
        await stopRecordingWeb();
      } else {
        await stopRecordingNative();
      }

      setRecordingState((prev) => ({
        ...prev,
        isRecording: false,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to stop recording';
      setRecordingState((prev) => ({
        ...prev,
        error: errorMessage,
        isRecording: false,
      }));
      if (onError) {
        onError(errorMessage);
      }
    }
  }, [isWeb, stopRecordingWeb, stopRecordingNative, onError]);

  // Send final chunk
  const sendFinalChunk = useCallback(async () => {
    if (!recordingState.sessionId || !recordingState.isRecording) {
      return;
    }

    try {
      let blob: Blob | null = null;

      if (isWeb) {
        // For web, use the chunks we've been collecting
        if (webChunksRef.current.length > 0) {
          const chunkBlob = new Blob(webChunksRef.current, { type: 'audio/webm' });
          if (chunkBlob.size > 0) {
            await processAndSendChunk(
              chunkBlob,
              recordingState.chunkIndex,
              recordingState.sessionId!
            );
            webChunksRef.current = []; // Clear after sending
          }
        }
        return;
      } else if (!isWeb && recordingRef.current) {
        const uri = recordingRef.current.getURI();
        if (uri) {
          const response = await fetch(uri);
          blob = await response.blob();
          if (blob && blob.size > 0) {
            await processAndSendChunk(blob, recordingState.chunkIndex, recordingState.sessionId);
          }
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send final chunk';
      console.error('Error sending final chunk:', error);
      if (onError) {
        onError(errorMessage);
      }
      throw error;
    }
  }, [recordingState, isWeb, processAndSendChunk, onError]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (chunkIntervalRef.current) {
        clearInterval(chunkIntervalRef.current);
      }
      if (isWeb) {
        stopRecordingWeb();
      } else {
        stopRecordingNative();
      }
    };
  }, [isWeb, stopRecordingWeb, stopRecordingNative]);

  return {
    recordingState,
    startRecording,
    stopRecording,
    sendFinalChunk,
  };
}
