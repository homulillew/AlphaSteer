import logging

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from typing import Optional, Tuple
from transformers.cache_utils import Cache
from utils.mask_utils import get_last_valid_token_index
from utils.local_steering_utils import (
    LocalProjectorFamily,
    compute_kernel_weights,
    compute_composite_steering_matrix,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Only CausalLM can be called from outside
__all__ = [
    'LocalAlphaLlamaForCausalLM',
]


class LocalAlphaLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Decoder layer with locally-gated projector steering (LocalAlphaSteer).

    Replaces AlphaSteer's single global projector with K locally-gated
    null-space projectors weighted by soft kernel assignments. When K=1,
    this degenerates exactly to AlphaSteer's global projector.
    """

    def __init__(self, config: LlamaConfig,
                 layer_idx: int,
                 local_family: Optional[LocalProjectorFamily] = None,
                 strength: float = 0.0
                 ):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.local_family = local_family
        self.strength = strength

    def set_steering_parameters(self,
        local_family: Optional[LocalProjectorFamily] = None,
        strength: float = 0.0,
        device: Optional[torch.device] = None):

        device = next(self.parameters()).device if device is None else device

        if local_family is not None:
            self.local_family = local_family.to(device)
        self.strength = strength

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass with locally-gated projector steering.

        The steering logic:
        1. Extract last-token hidden state per batch element
        2. Compute kernel weights w_k(h) for each of K clusters
        3. Form composite steering matrix M(h) = Σ_k w_k(h) * S_k
        4. Apply: h' = h + λ * (h @ M(h))
        5. Broadcast steering to all positions

        When K=1, this reduces to AlphaSteer:
            M(h) = w_1(h) * S_1 = S_1 = P @ Δ̃
            h' = h + λ * (h @ P @ Δ̃)
        """
        should_apply_steering = (
            hidden_states.shape[1] > 1
            and self.local_family is not None
            and self.strength != 0.0
        )

        if should_apply_steering:
            B, T, D = hidden_states.shape
            device = hidden_states.device

            # Get last valid token index (handles padding)
            last_idx = get_last_valid_token_index(
                attention_mask=attention_mask,
                seq_len=T,
                batch_size=B,
                device=device,
            )  # (B,)

            batch_idx = torch.arange(B, device=device)
            last_hidden = hidden_states[batch_idx, last_idx, :]  # (B, D)

            # Compute position-dependent composite steering matrix
            # M(h): (B, D, D) — different for each sample in the batch
            M_h = self.local_family.get_steering_matrix(last_hidden)  # (B, D, D)

            # Apply steering: h @ M(h) for each batch element
            # last_hidden: (B, D) -> (B, 1, D), M_h: (B, D, D)
            steering_vector = torch.bmm(
                last_hidden.unsqueeze(1), M_h
            ).squeeze(1) * self.strength  # (B, D)

            # Broadcast to all sequence positions
            steering_vector = steering_vector.unsqueeze(1)  # (B, 1, D)
            hidden_states = hidden_states + steering_vector

        residual = hidden_states  # resid_pre

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states  # resid_mid

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states  # resid_post

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class LocalAlphaLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LocalAlphaLlamaDecoderLayer(
                config=config,
                layer_idx=layer_idx,
            )
             for layer_idx in range(config.num_hidden_layers)]
        )

    def set_steering_parameters(
        self,
        local_families: Optional[dict] = None,
        strength: Optional[list] = None,
        device: Optional[torch.device] = None):
        """
        Set steering parameters for all layers.

        Parameters:
        - local_families: dict mapping layer_idx -> LocalProjectorFamily
                          (only steered layers need entries)
        - strength: list of floats, one per layer
        - device: target device
        """
        device = next(self.parameters()).device if device is None else device

        for layer_idx, layer in enumerate(self.layers):
            family = None
            if local_families is not None and layer_idx in local_families:
                family = local_families[layer_idx]

            layer.set_steering_parameters(
                local_family=family,
                strength=strength[layer_idx] if strength is not None else 0.0,
                device=device,
            )
            torch.cuda.empty_cache()

        self.print_steering_parameters()

    def print_steering_parameters(self):
        logger.info("LocalAlphaSteer Steering Parameters:")
        logger.info(f"{'Layer':<10}{'Strength':<20}{'K (clusters)':<15}{'Has Family'}")
        logger.info("=" * 65)
        for layer_idx, layer in enumerate(self.layers):
            strength_val = str(layer.strength)
            if layer.local_family is not None:
                k_val = str(layer.local_family.K)
                has_family = "Yes"
            else:
                k_val = "-"
                has_family = "No"
            logger.info(f"{layer_idx:<10}{strength_val:<20}{k_val:<15}{has_family}")


class LocalAlphaLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LocalAlphaLlamaModel(config=config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        local_families: Optional[dict] = None,
                        strength: Optional[list] = None,
                        **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.set_steering_parameters(local_families=local_families, strength=strength)
        return model

    def set_steering_parameters(
            self,
            local_families: Optional[dict] = None,
            strength: Optional[list] = None):

        device = next(self.parameters()).device
        self.model.set_steering_parameters(
            local_families=local_families,
            strength=strength,
            device=device,
        )
