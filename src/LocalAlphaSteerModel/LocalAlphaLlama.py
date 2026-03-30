import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from typing import Optional, Tuple
from transformers.cache_utils import Cache
from utils.mask_utils import get_last_valid_token_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


__all__ = ['LocalAlphaLlamaForCausalLM']


class LocalAlphaLlamaDecoderLayer(LlamaDecoderLayer):
    """
    A decoder layer that applies locally-gated projector steering.

    Instead of a single global steering matrix, this layer maintains K local
    steering matrices and K anchor points. At inference, the steering is
    computed as a position-dependent blend:

        s(h) = strength * (sum_k w_k(h) * S_k) @ h

    where w_k(h) = softmax(-||h - z_k||^2 / tau) are soft gating weights.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int,
                 anchors: Optional[torch.Tensor] = None,
                 steering_matrices: Optional[torch.Tensor] = None,
                 tau: float = 1.0,
                 strength: float = 0.0):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.anchors = anchors           # [K, d]
        self.steering_matrices = steering_matrices  # [K, d, d]
        self.tau = tau
        self.strength = strength

    def set_steering_parameters(self,
                                anchors: Optional[torch.Tensor] = None,
                                steering_matrices: Optional[torch.Tensor] = None,
                                tau: float = 1.0,
                                strength: float = 0.0,
                                device: Optional[torch.device] = None):
        device = next(self.parameters()).device if device is None else device

        if anchors is not None and steering_matrices is not None:
            self.anchors = anchors.to(device)
            self.steering_matrices = steering_matrices.to(device)
        self.tau = tau
        self.strength = strength

    def _compute_local_steering(self, h):
        """
        Compute locally-gated steering vector for activations h.

        Parameters:
        - h: [B, d] - Last-token hidden states

        Returns:
        - steering_vector: [B, d]
        """
        B, d = h.shape
        device = h.device

        # Compute soft gating weights: [B, K]
        # Compute squared distances directly (avoids sqrt in cdist)
        sq_dists = (h.unsqueeze(1) - self.anchors.unsqueeze(0)).pow(2).sum(-1)  # [B, K]
        logits = -sq_dists / self.tau
        weights = F.softmax(logits, dim=-1)  # [B, K]

        # Blend steering matrices and apply
        # steering_matrices: [K, d, d], weights: [B, K]
        blended_S = torch.einsum('bk,kij->bij', weights, self.steering_matrices)  # [B, d, d]
        steering_vector = torch.bmm(blended_S, h.unsqueeze(-1)).squeeze(-1)  # [B, d]

        return steering_vector * self.strength

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
        Steering is applied only during initial input processing (seq_len > 1).
        """
        should_apply_steering = (
            hidden_states.shape[1] > 1
            and self.anchors is not None
            and self.steering_matrices is not None
            and self.strength != 0.0
        )

        if should_apply_steering:
            if self.anchors.device != hidden_states.device:
                self.anchors = self.anchors.to(hidden_states.device)
                self.steering_matrices = self.steering_matrices.to(hidden_states.device)

            B, T, D = hidden_states.shape
            device = hidden_states.device

            # Get last valid token index
            last_idx = get_last_valid_token_index(
                attention_mask=attention_mask,
                seq_len=T,
                batch_size=B,
                device=device,
            )

            # Extract last-token hidden states
            batch_idx = torch.arange(B, device=device)
            last_hidden = hidden_states[batch_idx, last_idx, :]  # [B, D]

            # Compute locally-gated steering vector
            steering_vector = self._compute_local_steering(last_hidden)  # [B, D]

            # Reshape to match hidden_states dimensions and apply steering
            # (broadcast to all positions, consistent with AlphaSteer)
            hidden_states = hidden_states + steering_vector.unsqueeze(1)

        # Standard decoder layer forward pass
        residual = hidden_states

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
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

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
        local_steering_params: Optional[dict] = None,
        strength: Optional[list] = None,
        device: Optional[torch.device] = None):
        """
        Set locally-gated steering parameters for all layers.

        Parameters:
        - local_steering_params: dict with keys per layer index, each containing:
            - 'anchors': [K, d] tensor
            - 'steering_matrices': [K, d, d] tensor
            - 'tau': float
        - strength: list of floats, one per layer
        - device: Device to move parameters to
        """
        device = next(self.parameters()).device if device is None else device

        for layer_idx, layer in enumerate(self.layers):
            layer_anchors = None
            layer_S = None
            layer_tau = 1.0
            layer_strength = 0.0

            if local_steering_params is not None and layer_idx in local_steering_params:
                params = local_steering_params[layer_idx]
                layer_anchors = params['anchors'].to(device)
                layer_S = params['steering_matrices'].to(device)
                layer_tau = params['tau']

            if strength is not None:
                layer_strength = strength[layer_idx]

            layer.set_steering_parameters(
                anchors=layer_anchors,
                steering_matrices=layer_S,
                tau=layer_tau,
                strength=layer_strength,
                device=device,
            )
            torch.cuda.empty_cache()

        self.print_steering_parameters()

    def print_steering_parameters(self):
        logger.info("Local Steering Parameters:")
        logger.info(f"{'Layer':<10}{'Strength':<15}{'K':<8}{'Tau':<15}{'S_norm'}")
        logger.info("=" * 60)
        for layer_idx, layer in enumerate(self.layers):
            K = layer.anchors.shape[0] if layer.anchors is not None else 0
            tau = f"{layer.tau:.4f}" if layer.anchors is not None else "N/A"
            S_norm = (
                f"{layer.steering_matrices.norm():.6f}"
                if layer.steering_matrices is not None
                else "None"
            )
            logger.info(
                f"{layer_idx:<10}{layer.strength:<15}{K:<8}{tau:<15}{S_norm}"
            )


class LocalAlphaLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LocalAlphaLlamaModel(config=config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        local_steering_params: Optional[dict] = None,
                        strength: Optional[list] = None,
                        **kwargs):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model.set_steering_parameters(
            local_steering_params=local_steering_params,
            strength=strength
        )
        return model

    def set_steering_parameters(
            self,
            local_steering_params: Optional[dict] = None,
            strength: Optional[list] = None):
        device = next(self.parameters()).device
        self.model.set_steering_parameters(
            local_steering_params=local_steering_params,
            strength=strength,
            device=device
        )
