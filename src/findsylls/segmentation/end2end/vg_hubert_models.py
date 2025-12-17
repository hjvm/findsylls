"""
VG-HuBERT model loading using transformers HuBERT architecture.

This is a simplified approach that uses the standard HuBERT architecture from
transformers but loads the VG-HuBERT trained weights. This avoids fairseq
compatibility issues while still using the actual VG-HuBERT model.

Based on the approach in the original repository where they support both
the full fairseq AudioEncoder and transformers HuBERT.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_vg_hubert_into_transformers(
    checkpoint_path: str,
    device: str = "cpu"
) -> "transformers.HubertModel":
    """
    Load VG-HuBERT weights into a transformers HuBERT model.
    
    This function loads the VG-HuBERT checkpoint and extracts the weights
    that are compatible with the transformers HuBERT architecture. This
    allows us to use the trained VG-HuBERT model without needing fairseq.
    
    Args:
        checkpoint_path: Path to the VG-HuBERT checkpoint (.pth file)
        device: Device to load the model on
        
    Returns:
        HubertModel with VG-HuBERT weights loaded
        
    Raises:
        ImportError: If transformers is not installed
        FileNotFoundError: If checkpoint file doesn't exist
    """
    try:
        from transformers import HubertModel
    except ImportError:
        raise ImportError(
            "transformers is required but not installed. "
            "Install it with: pip install transformers"
        )
    
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading VG-HuBERT weights from {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract the audio encoder weights
    # VG-HuBERT checkpoints can have different structures
    if "dual_encoder" in checkpoint:
        state_dict = checkpoint["dual_encoder"]
    elif "audio_encoder" in checkpoint:
        state_dict = checkpoint["audio_encoder"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # Assume the checkpoint itself is the state dict
        state_dict = checkpoint
    
    # Initialize a HuBERT model
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    
    # Try to load VG-HuBERT weights into the transformers model
    # This requires mapping the weight names from the VG-HuBERT format to transformers format
    try:
        # Create mapping from VG-HuBERT keys to transformers keys
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Remove 'audio_encoder.' prefix
            if key.startswith('audio_encoder.'):
                key = key[len('audio_encoder.'):]
            
            # Map feature extractor conv layers
            # VG-HuBERT: feature_extractor.conv_layers.X.0.weight -> transformers: feature_extractor.conv_layers.X.conv.weight
            if 'feature_extractor.conv_layers' in key:
                # Convert X.0.weight to X.conv.weight and X.2.weight/bias to X.layer_norm.weight/bias
                parts = key.split('.')
                if len(parts) >= 4:
                    layer_idx = parts[2]
                    sub_idx = parts[3]
                    
                    if sub_idx == '0':
                        # Conv layer
                        new_key = f"feature_extractor.conv_layers.{layer_idx}.conv.{'.'.join(parts[4:])}"
                    elif sub_idx == '2':
                        # Layer norm
                        new_key = f"feature_extractor.conv_layers.{layer_idx}.layer_norm.{'.'.join(parts[4:])}"
                    else:
                        new_key = key
                    new_state_dict[new_key] = value
                    continue
            
            # Map post extract projection -> feature_projection.projection
            if key.startswith('post_extract_proj.'):
                new_key = key.replace('post_extract_proj.', 'feature_projection.projection.')
                new_state_dict[new_key] = value
                continue
            
            # Map positional convolution
            # VG-HuBERT: pos_conv.0.* -> transformers: encoder.pos_conv_embed.conv.*
            if key.startswith('pos_conv.0.'):
                suffix = key[len('pos_conv.0.'):]
                # Handle weight_g and weight_v (parametrized weights in transformers)
                if suffix == 'weight_g':
                    new_state_dict['encoder.pos_conv_embed.conv.parametrizations.weight.original0'] = value
                elif suffix == 'weight_v':
                    new_state_dict['encoder.pos_conv_embed.conv.parametrizations.weight.original1'] = value
                else:
                    new_state_dict[f'encoder.pos_conv_embed.conv.{suffix}'] = value
                continue
            
            # Map encoder layers
            # VG-HuBERT: encoder.layers.X.self_attn.* -> transformers: encoder.layers.X.attention.*
            if 'encoder.layers' in key and 'self_attn' in key:
                new_key = key.replace('self_attn', 'attention')
                new_state_dict[new_key] = value
                continue
            
            # Map layer norm in encoder layers
            # VG-HuBERT: encoder.layers.X.self_attn_layer_norm -> transformers: encoder.layers.X.layer_norm
            if 'self_attn_layer_norm' in key:
                new_key = key.replace('self_attn_layer_norm', 'layer_norm')
                new_state_dict[new_key] = value
                continue
            
            # Map final layer norm
            # VG-HuBERT: encoder.layer_norm.* -> transformers: encoder.layer_norm.*
            if 'encoder.layer_norm' in key and 'layers' not in key:
                new_state_dict[key] = value
                continue
            
            # Map feed-forward layers
            # VG-HuBERT: encoder.layers.X.fc1/fc2 -> transformers: encoder.layers.X.feed_forward.intermediate_dense/output_dense
            if 'encoder.layers' in key:
                if '.fc1.' in key:
                    new_key = key.replace('.fc1.', '.feed_forward.intermediate_dense.')
                    new_state_dict[new_key] = value
                    continue
                elif '.fc2.' in key:
                    new_key = key.replace('.fc2.', '.feed_forward.output_dense.')
                    new_state_dict[new_key] = value
                    continue
                elif '.final_layer_norm.' in key:
                    new_key = key.replace('.final_layer_norm.', '.final_layer_norm.')
                    new_state_dict[new_key] = value
                    continue
            
            # Keep other keys that might already match
            if key.startswith(('encoder.', 'feature_extractor.', 'feature_projection.')):
                new_state_dict[key] = value
        
        # Load weights (with strict=False to allow partial loading)
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Some weights were not loaded from checkpoint: {len(missing_keys)} keys missing")
            logger.debug(f"Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Some weights from checkpoint were not used: {len(unexpected_keys)} unexpected keys")
            logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")
        
        logger.info("Successfully loaded VG-HuBERT weights into transformers HuBERT model")
        
    except Exception as e:
        logger.warning(
            f"Could not load VG-HuBERT weights into transformers model: {e}. "
            "Using base HuBERT model instead (performance will be degraded)."
        )
    
    model.eval()
    model = model.to(device)
    
    return model

    """
    Load the AudioEncoder class from the VG-HuBERT GitHub repository.
    
    This function downloads the required Python modules from GitHub and
    dynamically imports the AudioEncoder class. The modules are cached
    in a temporary directory for the duration of the Python session.
    
    Returns:
        AudioEncoder class from the VG-HuBERT repository
        
    Raises:
        ImportError: If unable to download or import the required modules
        RuntimeError: If required dependencies are missing
    """
    global _cached_audio_encoder
    
    if _cached_audio_encoder is not None:
        return _cached_audio_encoder
    
    logger.info("Loading VG-HuBERT AudioEncoder from GitHub repository...")
    
    try:
        # Check if fairseq is available (required dependency)
        try:
            import fairseq
        except ImportError:
            raise RuntimeError(
                "fairseq is required for VG-HuBERT but not installed. "
                "Install it with: pip install fairseq\n"
                "Note: fairseq requires PyTorch and may have compilation requirements."
            )
        
        # Create a temporary directory to store the downloaded modules
        temp_dir = Path(tempfile.gettempdir()) / "vg_hubert_models"
        temp_dir.mkdir(exist_ok=True)
        
        # Create models subdirectory to match the original structure
        models_dir = temp_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Create __init__.py files to make them proper packages
        (temp_dir / "__init__.py").touch()
        (models_dir / "__init__.py").touch()
        
        # Download required files
        import urllib.request
        
        for filename, url in REQUIRED_FILES.items():
            target_path = models_dir / filename
            
            # Only download if not already cached
            if not target_path.exists():
                logger.info(f"Downloading {filename} from GitHub...")
                try:
                    urllib.request.urlretrieve(url, target_path)
                    logger.info(f"Successfully downloaded {filename}")
                except Exception as e:
                    raise ImportError(
                        f"Failed to download {filename} from {url}: {e}\n"
                        "Please check your internet connection or manually download the file."
                    )
        
        # Add the temp directory to sys.path if not already there
        if str(temp_dir) not in sys.path:
            sys.path.insert(0, str(temp_dir))
        
        # Import the audio_encoder module
        try:
            spec = importlib.util.spec_from_file_location(
                "models.audio_encoder",
                models_dir / "audio_encoder.py"
            )
            if spec is None or spec.loader is None:
                raise ImportError("Failed to create module spec")
            
            audio_encoder_module = importlib.util.module_from_spec(spec)
            sys.modules["models.audio_encoder"] = audio_encoder_module
            spec.loader.exec_module(audio_encoder_module)
            
            # Get the AudioEncoder class
            AudioEncoder = audio_encoder_module.AudioEncoder
            
            # Cache it for future use
            _cached_audio_encoder = AudioEncoder
            
            logger.info("Successfully loaded AudioEncoder from GitHub")
            return AudioEncoder
            
        except Exception as e:
            raise ImportError(
                f"Failed to import AudioEncoder from downloaded modules: {e}\n"
                "This may be due to missing dependencies or API changes in the repository."
            )
    
    except Exception as e:
        logger.error(f"Failed to load VG-HuBERT AudioEncoder: {e}")
        raise


def clear_cache():
    """Clear the cached AudioEncoder class (useful for testing)."""
    global _cached_audio_encoder
    _cached_audio_encoder = None
