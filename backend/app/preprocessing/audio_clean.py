"""
Audio cleaning and preprocessing utilities.
"""
import logging
import numpy as np
import librosa
from typing import Tuple, Optional

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logging.warning("noisereduce not available. Denoising will be skipped.")

logger = logging.getLogger(__name__)


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Supports: WAV, MP3, M4A, OGG, FLAC
    WebM files should be converted to WAV on the frontend before sending.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default 16000 Hz)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    
    Raises:
        ValueError: If file cannot be loaded or format is unsupported
        FileNotFoundError: If file doesn't exist
    """
    import os
    
    # Check if file exists
    if not os.path.exists(file_path):
        error_msg = f"Audio file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        error_msg = f"Audio file is empty: {file_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Attempting to load audio: {file_path} ({file_size} bytes)")
    
    try:
        # Try loading with librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        if len(audio) == 0:
            error_msg = f"Loaded audio is empty: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"✅ Loaded audio: {file_path}, duration: {len(audio)/sr:.2f}s, sr: {sr}Hz, samples: {len(audio)}")
        return audio, sr
        
    except FileNotFoundError:
        # Re-raise file not found errors
        raise
    except Exception as e:
        # Provide detailed error message
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else f"{error_type} occurred"
        
        # Check file extension for format hints
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.webm':
            detailed_error = (
                f"WebM format detected. librosa may not support WebM without ffmpeg. "
                f"Please convert to WAV on the frontend before sending. "
                f"Original error: {error_msg}"
            )
        elif file_ext in ['.mp3', '.m4a', '.aac']:
            detailed_error = (
                f"Audio format {file_ext} may require additional codecs. "
                f"Ensure ffmpeg is installed. Original error: {error_msg}"
            )
        else:
            detailed_error = (
                f"Failed to load audio file {file_path} (format: {file_ext or 'unknown'}). "
                f"Error type: {error_type}, Message: {error_msg}. "
                f"Supported formats: WAV, MP3, M4A, OGG, FLAC. "
                f"WebM files must be converted to WAV on the frontend."
            )
        
        logger.error(f"❌ {detailed_error}", exc_info=True)
        raise ValueError(detailed_error)


def trim_silence(audio: np.ndarray, sr: int, top_db: int = 40) -> np.ndarray:
    """
    Trim silence from the beginning and end of audio.
    
    Matches reference script: uses top_db=40 for consistency with training pipeline.
    
    Args:
        audio: Audio array
        sr: Sample rate
        top_db: Threshold in dB below reference for silence (default 40, matching reference)
    
    Returns:
        Trimmed audio array (or original if trimming fails)
    """
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        if trimmed.size == 0:
            # If trimming results in empty audio, return original
            logger.warning("Trimmed audio is empty, returning original")
            return audio
        logger.debug(f"Trimmed silence: {len(audio)/sr:.2f}s -> {len(trimmed)/sr:.2f}s")
        return trimmed
    except Exception as e:
        logger.warning(f"Error trimming silence: {e}, returning original audio")
        return audio


def denoise_audio(audio: np.ndarray, sr: int, stationary: bool = False) -> np.ndarray:
    """
    Reduce noise in audio using spectral gating.
    
    Args:
        audio: Audio array
        sr: Sample rate
        stationary: Whether noise is stationary
    
    Returns:
        Denoised audio array
    """
    if not NOISEREDUCE_AVAILABLE:
        logger.warning("noisereduce not available, skipping denoising")
        return audio
    
    try:
        denoised = nr.reduce_noise(y=audio, sr=sr, stationary=stationary)
        logger.debug("Applied noise reduction")
        return denoised
    except Exception as e:
        logger.warning(f"Error denoising audio: {e}, returning original audio")
        return audio


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio amplitude to target dB level.
    
    Args:
        audio: Audio array
        target_db: Target dB level (default -20 dB)
    
    Returns:
        Normalized audio array
    """
    try:
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            logger.warning("Audio is silent, skipping normalization")
            return audio
        
        # Calculate target RMS
        target_rms = 10 ** (target_db / 20.0)
        
        # Normalize
        normalized = audio * (target_rms / rms)
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val
        
        logger.debug(f"Normalized audio to {target_db:.1f} dB")
        return normalized
    except Exception as e:
        logger.warning(f"Error normalizing audio: {e}, returning original audio")
        return audio


def segment_audio(
    audio: np.ndarray, 
    sr: int, 
    segment_length: float = 3.0, 
    overlap: float = 0.5
) -> list[Tuple[np.ndarray, float]]:
    """
    Segment audio into fixed-length clips with overlap.
    
    Args:
        audio: Audio array
        sr: Sample rate
        segment_length: Length of each segment in seconds (default 3.0s)
        overlap: Overlap between segments in seconds (default 0.5s)
    
    Returns:
        List of tuples (segment_audio, start_time)
    """
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * sr)
    hop_samples = segment_samples - overlap_samples
    
    segments = []
    start_idx = 0
    
    while start_idx + segment_samples <= len(audio):
        segment = audio[start_idx:start_idx + segment_samples]
        start_time = start_idx / sr
        segments.append((segment, start_time))
        start_idx += hop_samples
    
    # Add final segment if there's remaining audio
    if start_idx < len(audio):
        segment = audio[start_idx:]
        # Pad if too short
        if len(segment) < segment_samples // 2:
            # Too short, merge with previous if exists
            if segments:
                logger.debug("Final segment too short, skipping")
            else:
                # Pad with zeros
                padded = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
                segments.append((padded, start_idx / sr))
        else:
            segments.append((segment, start_idx / sr))
    
    logger.info(f"Segmented audio into {len(segments)} segments of ~{segment_length}s")
    return segments


def preprocess_audio(
    file_path: str,
    target_sr: int = 16000,
    trim: bool = True,
    denoise: bool = True,
    normalize: bool = True,
    segment: bool = True,
    segment_length: float = 3.0,
    segment_overlap: float = 0.5
) -> Tuple[list[Tuple[np.ndarray, float]], int]:
    """
    Complete audio preprocessing pipeline.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        trim: Whether to trim silence
        denoise: Whether to denoise
        normalize: Whether to normalize
        segment: Whether to segment into clips
        segment_length: Length of segments in seconds
        segment_overlap: Overlap between segments in seconds
    
    Returns:
        Tuple of (list of (segment, start_time), sample_rate)
    """
    logger.info(f"Starting preprocessing for {file_path}")
    
    # Load audio
    audio, sr = load_audio(file_path, target_sr)
    
    # Trim silence
    if trim:
        audio = trim_silence(audio, sr)
    
    # Denoise
    if denoise:
        audio = denoise_audio(audio, sr)
    
    # Normalize
    if normalize:
        audio = normalize_audio(audio)
    
    # Segment
    if segment:
        segments = segment_audio(audio, sr, segment_length, segment_overlap)
    else:
        segments = [(audio, 0.0)]
    
    logger.info(f"Preprocessing complete: {len(segments)} segments")
    return segments, sr

