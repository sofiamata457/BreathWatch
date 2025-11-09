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
 * Convert AudioBuffer to WAV Blob
 */
function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  
  const length = buffer.length * numChannels * bytesPerSample;
  const arrayBuffer = new ArrayBuffer(44 + length);
  const view = new DataView(arrayBuffer);
  
  // WAV header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, 'data');
  view.setUint32(40, length, true);
  
  // Convert audio data
  let offset = 44;
  for (let i = 0; i < buffer.length; i++) {
    for (let channel = 0; channel < numChannels; channel++) {
      const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
  }
  
  return new Blob([arrayBuffer], { type: 'audio/wav' });
}

/**
 * Resample AudioBuffer to target sample rate
 */
async function resampleAudioBuffer(buffer: AudioBuffer, targetSampleRate: number): Promise<AudioBuffer> {
  const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
    sampleRate: targetSampleRate,
  });
  
  const ratio = buffer.sampleRate / targetSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const newBuffer = audioContext.createBuffer(buffer.numberOfChannels, newLength, targetSampleRate);
  
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const oldData = buffer.getChannelData(channel);
    const newData = newBuffer.getChannelData(channel);
    
    for (let i = 0; i < newLength; i++) {
      const index = Math.floor(i * ratio);
      newData[i] = oldData[index];
    }
  }
  
  return newBuffer;
}

/**
 * Convert audio blob to WAV format using Web Audio API
 * 
 * This function converts WebM/other formats to WAV on the client side
 * before sending to the backend, which expects WAV format.
 */
async function prepareAudioBlob(blob: Blob): Promise<Blob> {
  // Check if it's already WAV
  if (blob.type === 'audio/wav' || blob.type === 'audio/wave') {
    console.log('✅ Audio is already WAV format');
    return blob;
  }
  
  // Convert WebM/other formats to WAV using Web Audio API
  try {
    console.log(`🔄 Converting ${blob.type} to WAV...`);
    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    console.log(`📊 Audio buffer info:`, {
      duration: audioBuffer.duration.toFixed(2) + 's',
      sampleRate: audioBuffer.sampleRate,
      numberOfChannels: audioBuffer.numberOfChannels,
      length: audioBuffer.length,
    });
    
    // Resample to 16kHz if needed (backend expects 16kHz)
    let processedBuffer = audioBuffer;
    if (audioBuffer.sampleRate !== SAMPLE_RATE) {
      console.log(`🔄 Resampling from ${audioBuffer.sampleRate}Hz to ${SAMPLE_RATE}Hz...`);
      processedBuffer = await resampleAudioBuffer(audioBuffer, SAMPLE_RATE);
    }
    
    // Convert AudioBuffer to WAV
    const wavBlob = audioBufferToWav(processedBuffer);
    console.log(`✅ Converted ${blob.type} to WAV: ${(blob.size / 1024).toFixed(2)} KB -> ${(wavBlob.size / 1024).toFixed(2)} KB`);
    return wavBlob;
  } catch (error) {
    console.error('❌ Error converting audio to WAV:', error);
    throw new Error(`Failed to convert audio to WAV format: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
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
          console.log(`📦 Data available: ${(event.data.size / 1024).toFixed(2)} KB, total chunks: ${webChunksRef.current.length}`);
          
          // When we have a complete chunk (10 minutes), process it
          const timeSinceLastChunk = Date.now() - (lastChunkTimeRef.current || 0);
          if (timeSinceLastChunk >= CHUNK_DURATION_MS) {
            const chunkBlob = new Blob(webChunksRef.current, { type: 'audio/webm' });
            console.log(`⏰ 10 minutes elapsed, processing chunk ${chunkIndex}...`);
            await processAndSendChunk(chunkBlob, chunkIndex, sessionId);
            webChunksRef.current = []; // Clear chunks
            chunkIndex++;
            lastChunkTimeRef.current = Date.now();
          }
        } else {
          console.warn('⚠️ Data available event fired but data size is 0');
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
        console.log('🛑 Stopping MediaRecorder, requesting final data...');
        mediaRecorderRef.current.requestData();
        
        // Stop the recorder
        mediaRecorderRef.current.stop();
        
        // Wait a bit for ondataavailable to fire (the existing handler will collect the data)
        // This ensures any final data is captured before we resolve
        setTimeout(() => {
          const totalSize = webChunksRef.current.reduce((sum, chunk) => sum + chunk.size, 0);
          console.log(`🛑 MediaRecorder stopped. Total chunks: ${webChunksRef.current.length}, total size: ${(totalSize / 1024).toFixed(2)} KB`);
          resolve();
        }, 500); // Give time for final data event
      } else {
        resolve();
      }
      
      // Stop media tracks
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => {
          track.stop();
          console.log('🔇 Stopped media track');
        });
        mediaStreamRef.current = null;
      }
      
      // Don't null the recorder ref yet - we need it for sendFinalChunk
      // mediaRecorderRef.current = null;
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
    // Get sessionId from current state (don't rely on recordingState which might be stale)
    const currentSessionId = recordingState.sessionId;
    if (!currentSessionId) {
      console.warn('⚠️ No session ID available for final chunk');
      return;
    }

    try {
      let blob: Blob | null = null;

      if (isWeb) {
        // For web, we need to wait for MediaRecorder to provide all data
        // Request final data and wait for it
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
          // Request any remaining data
          mediaRecorderRef.current.requestData();
          
          // Wait a bit for ondataavailable to fire
          await new Promise((resolve) => setTimeout(resolve, 300));
        }

        // Collect all chunks we have (even if recording was very short)
        const chunksToSend = [...webChunksRef.current];
        console.log(`📤 Preparing final chunk: ${chunksToSend.length} blob chunks, total size: ${chunksToSend.reduce((sum, c) => sum + c.size, 0)} bytes`);
        
        if (chunksToSend.length > 0) {
          const chunkBlob = new Blob(chunksToSend, { type: 'audio/webm' });
          console.log(`📤 Final chunk blob size: ${(chunkBlob.size / 1024).toFixed(2)} KB`);
          
          if (chunkBlob.size > 0) {
            const currentChunkIndex = recordingState.chunkIndex;
            console.log(`📤 Sending final chunk (index ${currentChunkIndex}) to backend...`);
            await processAndSendChunk(
              chunkBlob,
              currentChunkIndex,
              currentSessionId
            );
            webChunksRef.current = []; // Clear after sending
            console.log(`✅ Final chunk sent successfully`);
          } else {
            console.warn('⚠️ Final chunk blob is empty (0 bytes)');
          }
        } else {
          console.warn('⚠️ No chunks collected for final chunk - recording may have been too short or no data was captured');
          // Still try to send an empty blob or create a minimal one to ensure session is created
          // But actually, we should still send something to ensure the session exists
          const emptyBlob = new Blob([], { type: 'audio/webm' });
          if (emptyBlob.size === 0) {
            console.warn('⚠️ Cannot send empty blob. Recording may have been too short.');
          }
        }
        return;
      } else if (!isWeb && recordingRef.current) {
        const uri = recordingRef.current.getURI();
        if (uri) {
          const response = await fetch(uri);
          blob = await response.blob();
          console.log(`📤 Final chunk (native): ${(blob.size / 1024).toFixed(2)} KB`);
          if (blob && blob.size > 0) {
            await processAndSendChunk(blob, recordingState.chunkIndex, currentSessionId);
            console.log(`✅ Final chunk sent successfully`);
          } else {
            console.warn('⚠️ Final chunk blob is empty (0 bytes)');
          }
        } else {
          console.warn('⚠️ No recording URI available for final chunk');
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send final chunk';
      console.error('❌ Error sending final chunk:', error);
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
