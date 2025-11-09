/**
 * Utility to temporarily store audio files
 */

import { Platform } from 'react-native';

// Store blob URLs in memory for access
const storedAudioFiles: Map<string, { url: string; blob: Blob; filename: string; timestamp: number }> = new Map();

/**
 * Save audio blob to temporary storage
 * For web: creates blob URL and stores in memory (can be downloaded if needed)
 * For native: saves to file system (would need expo-file-system)
 */
export async function saveAudioTemporarily(
  audioBlob: Blob,
  filename: string
): Promise<string> {
  if (Platform.OS === 'web') {
    // For web: create a blob URL and store it
    const url = URL.createObjectURL(audioBlob);
    const fileId = `audio_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Store in memory for later access
    storedAudioFiles.set(fileId, {
      url,
      blob: audioBlob,
      filename,
      timestamp: Date.now(),
    });
    
    // Log file info
    const sizeKB = (audioBlob.size / 1024).toFixed(2);
    console.log(`📁 Audio file temporarily stored: ${filename} (${sizeKB} KB)`);
    console.log(`   Blob URL: ${url}`);
    console.log(`   File ID: ${fileId}`);
    console.log(`   Access in console: window.getStoredAudio('${fileId}')`);
    
    // Make it accessible globally for debugging
    if (typeof window !== 'undefined') {
      (window as any).getStoredAudio = (id: string) => {
        const file = storedAudioFiles.get(id);
        if (file) {
          // Trigger download
          const link = document.createElement('a');
          link.href = file.url;
          link.download = file.filename;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          return file;
        }
        return null;
      };
      (window as any).listStoredAudio = () => {
        return Array.from(storedAudioFiles.entries()).map(([id, file]) => ({
          id,
          filename: file.filename,
          size: `${(file.blob.size / 1024).toFixed(2)} KB`,
          timestamp: new Date(file.timestamp).toISOString(),
        }));
      };
    }
    
    // Return the blob URL
    return url;
  } else {
    // For native: would use expo-file-system
    // For now, just log the info
    const sizeKB = (audioBlob.size / 1024).toFixed(2);
    console.log(`📁 Audio file would be saved: ${filename} (${sizeKB} KB)`);
    console.log('   Native file saving not yet implemented - would use expo-file-system');
    
    // Return a placeholder
    return `file://temp/${filename}`;
  }
}

/**
 * Clean up blob URL (for web)
 */
export function cleanupBlobUrl(url: string): void {
  if (Platform.OS === 'web' && url.startsWith('blob:')) {
    URL.revokeObjectURL(url);
  }
}

