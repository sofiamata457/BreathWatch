"""
Mobile-optimized feature extraction for 1-second log-Mel windows.
Matches CoughMultitaskCNN training pipeline specifications.
"""
import numpy as np
import librosa
from typing import Tuple, Optional
import logging
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


def extract_1s_logmel_window(
    audio: np.ndarray,
    sr: int = 16000,
    window_length: float = 1.0,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: float = 20.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Extract 1-second log-Mel spectrogram window matching CoughMultitaskCNN specs.
    
    Pipeline:
    1. Extract Mel spectrogram with exact training parameters
    2. Convert to dB using librosa.power_to_db
    3. Resize to [256, 256] using bilinear interpolation
    4. Normalize: (spec + 80.0) / 80.0, clip to [0, 1]
    5. Add channel dimension: [256, 256] → [1, 256, 256]
    
    Args:
        audio: Audio array (should be ~1 second at sr)
        sr: Sample rate (16kHz)
        window_length: Window length in seconds (1.0s)
        n_mels: Number of mel filter banks (80)
        n_fft: FFT window size (400 = 25ms at 16kHz)
        hop_length: Hop length (160 = 10ms at 16kHz)
        win_length: Window length (400)
        fmin: Minimum frequency (20 Hz)
        fmax: Maximum frequency (None = sr/2)
    
    Returns:
        Log-Mel spectrogram array [1, 256, 256] ready for model input
    """
    # Ensure audio is at least window_length
    min_samples = int(window_length * sr)
    if len(audio) < min_samples:
        # Pad with zeros
        audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
    elif len(audio) > min_samples:
        # Truncate to window_length
        audio = audio[:min_samples]
    
    # Extract Mel spectrogram with exact training parameters
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax
    )
    
    # Convert to dB using librosa.power_to_db (matches training)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Shape: [80, T] where T depends on hop_length
    
    # Resize [80, T] → [256, 256] using bilinear interpolation
    if mel_spec_db.shape != (256, 256):
        zoom_factors = (256 / mel_spec_db.shape[0], 256 / mel_spec_db.shape[1])
        mel_spec_db = zoom(mel_spec_db, zoom_factors, order=1)  # order=1 is bilinear
    
    # Normalize exactly like training: (spec + 80.0) / 80.0
    mel_spec_normalized = (mel_spec_db + 80.0) / 80.0
    
    # Clip to [0, 1]
    mel_spec_normalized = np.clip(mel_spec_normalized, 0.0, 1.0)
    
    # Add channel dimension: [256, 256] → [1, 256, 256]
    mel_spec_normalized = np.expand_dims(mel_spec_normalized, axis=0)
    
    logger.debug(f"Extracted 1s log-Mel window: shape {mel_spec_normalized.shape}")
    return mel_spec_normalized


def segment_audio_1s_windows(
    audio: np.ndarray,
    sr: int = 16000,
    window_length: float = 1.0,
    stride: float = 0.25
) -> list[Tuple[np.ndarray, float]]:
    """
    Segment audio into 1-second windows with 0.25s stride (0.75s overlap).
    
    Matches training pipeline: 1.0s tiles with 0.25s between tile starts.
    
    Args:
        audio: Audio array
        sr: Sample rate
        window_length: Window length in seconds (1.0s)
        stride: Stride between window starts in seconds (0.25s = 0.75s overlap)
    
    Returns:
        List of tuples (window_audio, start_time)
    """
    window_samples = int(window_length * sr)
    stride_samples = int(stride * sr)
    
    windows = []
    start_idx = 0
    
    while start_idx + window_samples <= len(audio):
        window = audio[start_idx:start_idx + window_samples]
        start_time = start_idx / sr
        windows.append((window, start_time))
        start_idx += stride_samples
    
    # Add final window if there's remaining audio (pad if needed)
    if start_idx < len(audio):
        window = audio[start_idx:]
        if len(window) >= window_samples // 2:  # Only if at least half a window
            # Pad to full window
            window = np.pad(window, (0, window_samples - len(window)), mode='constant')
            windows.append((window, start_idx / sr))
    
    return windows


def prepare_mobile_features(
    audio: np.ndarray,
    sr: int = 16000,
    window_length: float = 1.0,
    n_mels: int = 80
) -> np.ndarray:
    """
    Prepare features for CoughMultitaskCNN model input.
    
    Returns features in shape [1, 256, 256] matching training pipeline.
    
    Args:
        audio: Audio array (1 second)
        sr: Sample rate (16kHz)
        window_length: Window length (1.0s)
        n_mels: Number of mel bins (80)
    
    Returns:
        Feature array ready for model: [1, 256, 256]
    """
    # Extract 1s log-Mel window (already returns [1, 256, 256])
    mel_spec = extract_1s_logmel_window(
        audio, sr=sr, window_length=window_length, n_mels=n_mels
    )
    
    # Should already be [1, 256, 256] from extract_1s_logmel_window
    # But ensure it's correct shape
    if len(mel_spec.shape) == 2:
        mel_spec = np.expand_dims(mel_spec, axis=0)  # Add channel if missing
    
    return mel_spec

