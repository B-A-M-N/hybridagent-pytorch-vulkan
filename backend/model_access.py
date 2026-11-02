"""Model access helper for HybridAgent and Qwen3 models."""
import torch
from typing import Tuple, Any


def get_qwen3_model(model: Any) -> torch.nn.Module:
    """
    Extract the Qwen3Model decoder from various HybridAgent/Hugging Face wrappers.
    
    Args:
        model: HybridAgentModel, Qwen3ForCausalLM, or Qwen3Model
        
    Returns:
        Qwen3Model with .layers attribute
        
    Raises:
        ValueError: If model type is not recognized
    """
    from hybrid_agent.model import HybridAgentModel
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model
    
    if isinstance(model, HybridAgentModel):
        # HybridAgentModel has .backbone which is Qwen3ForCausalLM
        # and .backbone.model is Qwen3Model
        if hasattr(model.backbone, 'model'):
            return model.backbone.model
        raise ValueError(f"HybridAgentModel backbone missing .model: {type(model.backbone)}")
    
    if isinstance(model, Qwen3ForCausalLM):
        # Qwen3ForCausalLM has .model which is Qwen3Model
        if hasattr(model, 'model'):
            return model.model
        raise ValueError("Qwen3ForCausalLM missing .model attribute")
    
    if isinstance(model, Qwen3Model):
        return model
    
    # Fallback: try common attributes
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'model'):
        return model.backbone.model
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model
    
    raise ValueError(
        f"Unsupported model type: {type(model)}. "
        f"Expected HybridAgentModel, Qwen3ForCausalLM, or Qwen3Model"
    )


def get_qwen3_layers(model: Any) -> torch.nn.ModuleList:
    """Get the layers ModuleList from a Qwen3 model."""
    qwen3_model = get_qwen3_model(model)
    if not hasattr(qwen3_model, 'layers'):
        raise ValueError(f"Qwen3Model {type(qwen3_model)} has no .layers attribute")
    return qwen3_model.layers


def get_model_config(model: Any):
    """Get config from various model wrappers."""
    if hasattr(model, 'config'):
        return model.config
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'config'):
        return model.backbone.config
    raise ValueError(f"Cannot find config in {type(model)}")


def create_hybrid_backbone(qwen3_model, placement):
    """Create HybridBackbone from Qwen3Model with placement."""
    from backend.hybrid_offload import HybridBackbone
    return HybridBackbone(qwen3_model, placement)