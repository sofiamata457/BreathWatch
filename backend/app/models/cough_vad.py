"""
Cough model loader and inference with attribute probabilities.
Outputs: p_cough + 5 attribute probabilities (wet, stridor, choking, congestion, wheezing_selfreport).
Optimized for mobile deployment with 1-second log-Mel windows.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SimpleCoughModel(nn.Module):
    """Simple CNN model for cough detection with attributes."""
    def __init__(self, num_outputs=6):
        super().__init__()
        # CNN architecture matching the actual model structure (16->32->64 channels)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_outputs)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)


class CoughVAD:
    """Cough model with attribute probabilities."""
    
    def __init__(self, model_path: Optional[str] = None, model_class: Optional[callable] = None):
        """
        Initialize cough model.
        
        Args:
            model_path: Path to PyTorch model file (.pt). If None, looks in assets folder.
            model_class: Optional PyTorch model class to instantiate. Required if loading state dict.
        """
        self.model = None
        self.model_path = model_path
        self.model_class = model_class
        self.input_shape = (1, 256, 256)  # Matches CoughMultitaskCNN training specs
        self.num_outputs = 6  # p_cough + 5 attributes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self) -> None:
        """Load the PyTorch model."""
        if self.model_path is None:
            # Try to find model in models folder first, then assets folder
            models_path = Path(__file__).parent
            model_path = models_path / "cough_multitask.pt"
            if not model_path.exists():
                # Fallback to assets folder
                assets_path = Path(__file__).parent.parent.parent.parent / "assets"
                model_path = assets_path / "cough_model.pt"
                if not model_path.exists():
                    logger.warning(
                        f"Cough model not found at {models_path / 'cough_multitask.pt'} or {assets_path / 'cough_model.pt'}. "
                        "Using mock predictions. Place a trained PyTorch model at one of these locations."
                    )
                    self.model = None
                    return
            self.model_path = str(model_path)
        
        try:
            # Load PyTorch model
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Check if it's a full model or just state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # State dict format
                    if self.model_class is None:
                        raise ValueError("model_class required when loading state dict. Provide a callable that returns the model.")
                    self.model = self.model_class()  # Call the factory function
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    if self.model_class is None:
                        raise ValueError("model_class required when loading state dict")
                    self.model = self.model_class()  # Call the factory function
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume it's a state dict directly
                    if self.model_class is None:
                        # Try to use default architecture
                        logger.info("No model_class provided, using default SimpleCoughModel architecture")
                        self.model = SimpleCoughModel(num_outputs=self.num_outputs)
                    else:
                        self.model = self.model_class()  # Call the factory function
                    # Try to load state dict, ignoring extra/missing keys
                    try:
                        self.model.load_state_dict(checkpoint, strict=False)
                        logger.info("Loaded state dict (some keys may be missing/extra)")
                    except Exception as e:
                        logger.warning(f"Could not load state dict strictly: {e}. Trying to match keys...")
                        # Try to match keys manually
                        model_dict = self.model.state_dict()
                        matched_dict = {}
                        for k, v in checkpoint.items():
                            if k in model_dict and model_dict[k].shape == v.shape:
                                matched_dict[k] = v
                        if matched_dict:
                            model_dict.update(matched_dict)
                            self.model.load_state_dict(model_dict, strict=False)
                            logger.info(f"Loaded {len(matched_dict)}/{len(checkpoint)} keys from state dict")
                        else:
                            raise ValueError(f"Could not match any keys from state dict. Available keys: {list(checkpoint.keys())[:5]}...")
            else:
                # Assume it's a full model
                self.model = checkpoint
            
            # Set to eval mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Move to device
            self.model = self.model.to(self.device)
            
            logger.info(f"Loaded cough model from {self.model_path}")
            logger.info(f"Model device: {self.device}, outputs: {self.num_outputs}")
        except Exception as e:
            logger.error(f"Error loading cough VAD model: {e}", exc_info=True)
            self.model = None
    
    def predict(self, features: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """
        Predict cough and attribute probabilities.
        
        Args:
            features: Feature array (1s log-Mel spectrogram)
        
        Returns:
            Tuple of (p_cough, p_attr_wet, p_attr_stridor, p_attr_choking, p_attr_congestion, p_attr_wheezing_selfreport)
            All values in [0, 1]
        """
        if self.model is None:
            # Mock prediction for development
            logger.warning("Using mock cough model prediction (model not loaded)")
            return self._mock_predict(features)
        
        try:
            # Prepare input
            input_tensor = self._prepare_input(features)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Handle different output formats
            # CoughMultitaskCNN returns a tuple: (logits_a, logits_b, ...)
            # logits_a is [batch, 2] for binary cough classification
            logger.info(f"🔍 Model output type: {type(output)}, is tuple/list: {isinstance(output, (list, tuple))}")
            if isinstance(output, (list, tuple)):
                logger.info(f"🔍 Model output tuple length: {len(output)}")
                logits_a = output[0]  # First output is cough binary classification
                logger.info(f"🔍 First output (logits_a) type: {type(logits_a)}, shape: {logits_a.shape if hasattr(logits_a, 'shape') else 'unknown'}")
                if len(output) > 1:
                    logger.info(f"🔍 Second output (probs_b) type: {type(output[1])}, shape: {output[1].shape if hasattr(output[1], 'shape') else 'unknown'}, values: {output[1]}")
                
                # Check if logits_a is 2D [batch, 2] - apply softmax for binary classification
                if isinstance(logits_a, torch.Tensor):
                    logits_a_np = logits_a.cpu().numpy()
                else:
                    logits_a_np = np.array(logits_a)
                
                # If 2D with shape [batch, 2], apply softmax and extract cough probability
                # This matches the reference script: torch.softmax(logits_a, dim=1)[:, 1]
                if len(logits_a_np.shape) == 2 and logits_a_np.shape[1] == 2:
                    # Apply softmax: exp(x) / sum(exp(x)) for each row (dim=1)
                    exp_logits = np.exp(logits_a_np - np.max(logits_a_np, axis=1, keepdims=True))  # Numerical stability
                    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                    p_cough_from_softmax = float(softmax_probs[0, 1])  # Take cough probability (second column, first row)
                    logger.info(f"✅ Applied softmax to binary logits [batch, 2]: {logits_a_np[0]} -> {softmax_probs[0]}, p_cough={p_cough_from_softmax:.3f}")
                    
                    # For attributes, check if there are more outputs in the tuple
                    if len(output) > 1:
                        # Second output is probs_b from CoughMultitaskCNN - should already have sigmoid applied!
                        attr_output = output[1]
                        if isinstance(attr_output, torch.Tensor):
                            attr_array = attr_output.cpu().numpy()
                        else:
                            attr_array = np.array(attr_output)
                        
                        logger.info(f"🔍 Raw attribute output shape: {attr_array.shape}, values: {attr_array}")
                        
                        # Handle 2D [batch, 5] or 1D [5]
                        if len(attr_array.shape) == 2:
                            attr_raw = attr_array[0]  # Take first row
                        else:
                            attr_raw = attr_array.flatten()[:5]
                        
                        logger.info(f"🔍 Extracted attribute raw values: {attr_raw}")
                        logger.info(f"🔍 Min: {np.min(attr_raw):.3f}, Max: {np.max(attr_raw):.3f}, Mean: {np.mean(attr_raw):.3f}")
                        
                        # Check if values are in valid probability range [0, 1]
                        # If any value > 1, they're logits and need sigmoid
                        # ALSO check if all values are exactly 1.0 (which would be suspicious)
                        if np.allclose(attr_raw, 1.0, atol=0.01):
                            logger.error(f"⚠️⚠️⚠️ ALL ATTRIBUTE VALUES ARE 1.0! This is suspicious. Raw values: {attr_raw}")
                            logger.error(f"   This suggests the model might not be applying sigmoid correctly, or we're reading the wrong output.")
                            # Try to see if there's a different output or if we need to process differently
                            # For now, use the values as-is but log the issue
                            attr_probs = attr_raw
                        elif np.any(attr_raw > 1.0) or np.any(attr_raw < 0.0):
                            logger.warning(f"⚠️ Attribute values outside [0,1] range (likely logits): {attr_raw}, applying sigmoid")
                            attr_probs = 1 / (1 + np.exp(-np.clip(attr_raw, -500, 500)))
                            logger.info(f"✅ Applied sigmoid to attributes: {attr_raw} -> {attr_probs}")
                        else:
                            # Already probabilities
                            attr_probs = attr_raw
                            logger.info(f"✅ Using attribute probabilities from model (already in [0,1] range): {attr_probs}")
                    else:
                        # No separate attribute output - might be in the same tensor or model doesn't output them
                        # Check if logits_a has more columns (e.g., [batch, 7] = 2 cough + 5 attributes)
                        if logits_a_np.shape[1] >= 7:
                            # Extract attributes from columns 2-6
                            attr_logits = logits_a_np[0, 2:7]
                            attr_probs = 1 / (1 + np.exp(-np.clip(attr_logits, -500, 500)))
                            logger.info(f"✅ Extracted attributes from same tensor, applied sigmoid: {attr_logits} -> {attr_probs}")
                        else:
                            # No attributes available
                            attr_probs = np.zeros(5)
                            logger.warning("⚠️ No attribute output found in model, using zeros")
                    
                    # Return the processed values - CLAMP to [0, 1] to ensure valid probabilities
                    result = (
                        p_cough_from_softmax,
                        float(np.clip(attr_probs[0], 0.0, 1.0)) if len(attr_probs) > 0 else 0.0,
                        float(np.clip(attr_probs[1], 0.0, 1.0)) if len(attr_probs) > 1 else 0.0,
                        float(np.clip(attr_probs[2], 0.0, 1.0)) if len(attr_probs) > 2 else 0.0,
                        float(np.clip(attr_probs[3], 0.0, 1.0)) if len(attr_probs) > 3 else 0.0,
                        float(np.clip(attr_probs[4], 0.0, 1.0)) if len(attr_probs) > 4 else 0.0,
                    )
                    logger.info(f"✅ Final attribute probabilities (clamped to [0,1]): wet={result[1]:.3f}, stridor={result[2]:.3f}, choking={result[3]:.3f}, congestion={result[4]:.3f}, wheezing={result[5]:.3f}")
                    return result
                elif len(logits_a_np.shape) == 2 and logits_a_np.shape[1] >= 6:
                    # Single tensor with [batch, 6+] - first 2 are cough logits, rest are attributes
                    logger.info(f"Model output is [batch, {logits_a_np.shape[1]}] - treating as combined output")
                    # Apply softmax to first 2 columns (cough)
                    cough_logits = logits_a_np[0, :2]
                    exp_cough = np.exp(cough_logits - np.max(cough_logits))
                    cough_probs = exp_cough / np.sum(exp_cough)
                    p_cough_from_softmax = float(cough_probs[1])
                    
                    # Apply sigmoid to remaining columns (attributes)
                    if logits_a_np.shape[1] >= 7:
                        attr_logits = logits_a_np[0, 2:7]
                    else:
                        attr_logits = logits_a_np[0, 2:6] if logits_a_np.shape[1] >= 6 else np.zeros(5)
                        attr_logits = np.pad(attr_logits, (0, 5 - len(attr_logits)), mode='constant')
                    attr_probs = 1 / (1 + np.exp(-np.clip(attr_logits, -500, 500)))
                    
                    logger.info(f"✅ Processed combined output: p_cough={p_cough_from_softmax:.3f}, attributes={attr_probs}")
                    return (
                        p_cough_from_softmax,
                        float(attr_probs[0]),
                        float(attr_probs[1]),
                        float(attr_probs[2]),
                        float(attr_probs[3]),
                        float(attr_probs[4]),
                    )
                else:
                    # Not 2D [batch, 2] or [batch, 6+], treat as single output and flatten
                    output_array = logits_a_np.flatten()
                    logger.debug(f"Model output is not standard format, flattening: shape={logits_a_np.shape}")
            else:
                # Single output, not a tuple
                if isinstance(output, torch.Tensor):
                    output_array = output.cpu().numpy()
                else:
                    output_array = np.array(output)
                
                # Flatten if needed
                if len(output_array.shape) > 1:
                    output_array = output_array.flatten()
            
            # Ensure we have 6 outputs
            if len(output_array) < self.num_outputs:
                logger.warning(f"Model output has {len(output_array)} values, expected {self.num_outputs}")
                output_array = np.pad(output_array, (0, self.num_outputs - len(output_array)), mode='constant')
            elif len(output_array) > self.num_outputs:
                output_array = output_array[:self.num_outputs]
            
            # Log raw model output for debugging
            logger.info(f"🔍 Raw model output (before processing): shape={output_array.shape if hasattr(output_array, 'shape') else 'scalar'}, values={output_array}")
            
            # Check if output looks like logits (values outside [0,1] range or very large/small)
            max_abs_value = np.max(np.abs(output_array))
            needs_processing = max_abs_value > 5.0 or np.any(output_array < -2.0) or np.any(output_array > 2.0)
            
            if needs_processing:
                logger.debug(f"Model output appears to be logits (max_abs={max_abs_value:.3f}), applying transformations...")
                
                # For CoughMultitaskCNN, the first output is binary classification logits (no_cough, cough)
                # We need to apply softmax and take the second value (cough probability)
                # The other 5 outputs are attribute logits that need sigmoid
                
                # Process first output (cough): apply softmax to binary logits
                # If we have 2 values for cough, use softmax; otherwise assume single sigmoid output
                if len(output_array) >= 2:
                    # First two values might be [no_cough_logit, cough_logit] - apply softmax
                    cough_logits = output_array[:2]
                    cough_probs = np.exp(cough_logits) / np.sum(np.exp(cough_logits))
                    p_cough = float(cough_probs[1])  # Take cough probability (second value)
                    logger.debug(f"Applied softmax to cough logits: {cough_logits} -> {cough_probs}, p_cough={p_cough:.3f}")
                    
                    # Remaining outputs are attribute logits - apply sigmoid
                    attr_logits = output_array[2:7] if len(output_array) >= 7 else output_array[1:6]
                    attr_probs = 1 / (1 + np.exp(-attr_logits))
                    p_attr_wet = float(attr_probs[0]) if len(attr_probs) > 0 else 0.0
                    p_attr_stridor = float(attr_probs[1]) if len(attr_probs) > 1 else 0.0
                    p_attr_choking = float(attr_probs[2]) if len(attr_probs) > 2 else 0.0
                    p_attr_congestion = float(attr_probs[3]) if len(attr_probs) > 3 else 0.0
                    p_attr_wheezing_selfreport = float(attr_probs[4]) if len(attr_probs) > 4 else 0.0
                else:
                    # Fallback: assume all outputs need sigmoid
                    logger.warning("Unexpected output shape, applying sigmoid to all outputs")
                    output_array = 1 / (1 + np.exp(-output_array))
                    p_cough = float(np.clip(output_array[0], 0.0, 1.0))
                    p_attr_wet = float(np.clip(output_array[1], 0.0, 1.0)) if len(output_array) > 1 else 0.0
                    p_attr_stridor = float(np.clip(output_array[2], 0.0, 1.0)) if len(output_array) > 2 else 0.0
                    p_attr_choking = float(np.clip(output_array[3], 0.0, 1.0)) if len(output_array) > 3 else 0.0
                    p_attr_congestion = float(np.clip(output_array[4], 0.0, 1.0)) if len(output_array) > 4 else 0.0
                    p_attr_wheezing_selfreport = float(np.clip(output_array[5], 0.0, 1.0)) if len(output_array) > 5 else 0.0
            else:
                # Already probabilities
                logger.debug(f"Model output appears to be probabilities (max_abs={max_abs_value:.3f})")
                p_cough = float(np.clip(output_array[0], 0.0, 1.0))
                p_attr_wet = float(np.clip(output_array[1], 0.0, 1.0)) if len(output_array) > 1 else 0.0
                p_attr_stridor = float(np.clip(output_array[2], 0.0, 1.0)) if len(output_array) > 2 else 0.0
                p_attr_choking = float(np.clip(output_array[3], 0.0, 1.0)) if len(output_array) > 3 else 0.0
                p_attr_congestion = float(np.clip(output_array[4], 0.0, 1.0)) if len(output_array) > 4 else 0.0
                p_attr_wheezing_selfreport = float(np.clip(output_array[5], 0.0, 1.0)) if len(output_array) > 5 else 0.0
            
            logger.debug(f"Cough model prediction: cough={p_cough:.3f}, wet={p_attr_wet:.3f}, stridor={p_attr_stridor:.3f}, choking={p_attr_choking:.3f}, congestion={p_attr_congestion:.3f}, wheezing={p_attr_wheezing_selfreport:.3f}")
            return (p_cough, p_attr_wet, p_attr_stridor, p_attr_choking, p_attr_congestion, p_attr_wheezing_selfreport)
            
        except Exception as e:
            logger.error(f"Error in cough model prediction: {e}", exc_info=True)
            return self._mock_predict(features)
    
    def _prepare_input(self, features: np.ndarray) -> torch.Tensor:
        """
        Prepare features for model input.
        
        Expects features in shape [1, 256, 256] from prepare_mobile_features.
        Converts to [batch_size, 1, 256, 256] for model.
        """
        # Features should already be [1, 256, 256] from prepare_mobile_features
        # But handle different input shapes gracefully
        
        # If 2D, add channel dimension
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        
        # If shape is [1, H, W] but not [1, 256, 256], resize
        if len(features.shape) == 3 and features.shape[1:] != self.input_shape[1:]:
            from scipy.ndimage import zoom
            current_shape = features.shape[1:]
            zoom_factors = (
                self.input_shape[1] / current_shape[0],  # Height
                self.input_shape[2] / current_shape[1]   # Width
            )
            features = zoom(features, (1,) + zoom_factors, order=1)
        
        # Add batch dimension if missing (should be [batch, 1, 256, 256])
        if len(features.shape) == 3:
            features = np.expand_dims(features, axis=0)
        
        # Ensure shape is [batch, 1, 256, 256]
        if features.shape[1:] != self.input_shape:
            logger.warning(f"Feature shape {features.shape} doesn't match expected {self.input_shape}, attempting resize")
            from scipy.ndimage import zoom
            if len(features.shape) == 4:
                # [batch, channels, H, W]
                zoom_factors = (
                    1,  # batch
                    1,  # channels
                    self.input_shape[1] / features.shape[2],  # height
                    self.input_shape[2] / features.shape[3]   # width
                )
                features = zoom(features, zoom_factors, order=1)
        
        # Convert to torch tensor
        input_tensor = torch.from_numpy(features.astype(np.float32))
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor
    
    def _mock_predict(self, features: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """Mock prediction for development/testing."""
        # Simple heuristic: if audio has high energy in mid frequencies, might be cough
        if len(features.shape) >= 2:
            mid_freq_energy = np.mean(features[:, features.shape[1]//4:3*features.shape[1]//4])
            p_cough = min(0.8, max(0.2, mid_freq_energy))
        else:
            p_cough = 0.5
        
        # Mock attribute probabilities (lower than cough probability)
        p_attr_wet = p_cough * 0.6
        p_attr_stridor = p_cough * 0.3
        p_attr_choking = p_cough * 0.2
        p_attr_congestion = p_cough * 0.5
        p_attr_wheezing_selfreport = p_cough * 0.4
        
        return (p_cough, p_attr_wet, p_attr_stridor, p_attr_choking, p_attr_congestion, p_attr_wheezing_selfreport)
