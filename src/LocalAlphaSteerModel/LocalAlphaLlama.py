import logging

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.cache_utils import Cache

from typing import Optional, Tuple
from utils.mask_utils import get_last_valid_token_index
from utils.local_steering_utils import compute_local_steering_vector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

__all__ = ['LocalAlphaLlamaForCausalLM']


class LocalAlphaLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Llama decoder layer with LocalAlphaSteer soft-gated local projectors.

    At each steered layer the update is:
        h' = h + λ · (h @ M(h))
    where
        M(h) = Σ_k  w_k(h) · M_k
        w_k(h) = softmax( cosine_sim(h, c_k) / τ )

    K=1 reduces exactly to AlphaSteer.
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        centroids: Optional[torch.Tensor] = None,
        local_steering_matrices: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        strength: float = 0.0,
    ):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.strength = strength
        self.temperature = temperature

        device = next(self.parameters()).device
        self.centroids = centroids.to(device) if centroids is not None else None
        self.local_steering_matrices = (
            local_steering_matrices.to(device)
            if local_steering_matrices is not None
            else None
        )

    def set_steering_parameters(
        self,
        centroids: Optional[torch.Tensor] = None,
        local_steering_matrices: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        strength: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        device = next(self.parameters()).device if device is None else device

        if centroids is not None and torch.any(centroids):
            self.centroids = centroids.to(device)
        if local_steering_matrices is not None and torch.any(local_steering_matrices):
            self.local_steering_matrices = local_steering_matrices.to(device)
        self.temperature = temperature
        self.strength = strength

    @property
    def _should_apply_steering(self) -> bool:
        return (
            self.centroids is not None
            and self.local_steering_matrices is not None
            and self.strength != 0.0
        )

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

        if hidden_states.shape[1] > 1 and self._should_apply_steering:
            B, T, D = hidden_states.shape
            device = hidden_states.device

            if self.centroids.device != device:
                self.centroids = self.centroids.to(device)
            if self.local_steering_matrices.device != device:
                self.local_steering_matrices = self.local_steering_matrices.to(device)

            last_idx = get_last_valid_token_index(
                attention_mask=attention_mask,
                seq_len=T,
                batch_size=B,
                device=device,
            )
            batch_idx = torch.arange(B, device=device)
            last_hidden = hidden_states[batch_idx, last_idx, :]  # (B, D)

            steering_vector = compute_local_steering_vector(
                last_hidden.to(self.centroids.dtype),
                self.centroids,
                self.local_steering_matrices,
                temperature=self.temperature,
            ).to(hidden_states.dtype) * self.strength  # (B, D)

            hidden_states = hidden_states + steering_vector.unsqueeze(1)

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
            [LocalAlphaLlamaDecoderLayer(config=config, layer_idx=i)
             for i in range(config.num_hidden_layers)]
        )

    def set_steering_parameters(
        self,
        centroids: Optional[torch.Tensor] = None,
        local_steering_matrices: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        strength: Optional[list] = None,
        device: Optional[torch.device] = None,
    ):
        device = next(self.parameters()).device if device is None else device

        for layer_idx, layer in enumerate(self.layers):
            layer_centroids = (
                centroids[layer_idx] if centroids is not None else None
            )
            layer_matrices = (
                local_steering_matrices[layer_idx]
                if local_steering_matrices is not None
                else None
            )
            layer_strength = (
                strength[layer_idx] if strength is not None else 0.0
            )
            layer.set_steering_parameters(
                centroids=layer_centroids,
                local_steering_matrices=layer_matrices,
                temperature=temperature,
                strength=layer_strength,
                device=device,
            )
            torch.cuda.empty_cache()

        self._print_steering_parameters()

    def _print_steering_parameters(self):
        logger.info("LocalAlphaSteer Parameters:")
        logger.info(f"{'Layer':<10}{'Strength':<12}{'K':<6}{'Temperature'}")
        logger.info("=" * 50)
        for layer_idx, layer in enumerate(self.layers):
            K = layer.centroids.shape[0] if layer.centroids is not None else 0
            logger.info(
                f"{layer_idx:<10}{layer.strength:<12}{K:<6}{layer.temperature}"
            )


class LocalAlphaLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LocalAlphaLlamaModel(config=config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        centroids: Optional[torch.Tensor] = None,
        local_steering_matrices: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        strength: Optional[list] = None,
        **kwargs,
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model.set_steering_parameters(
            centroids=centroids,
            local_steering_matrices=local_steering_matrices,
            temperature=temperature,
            strength=strength,
        )
        return model

    def set_steering_parameters(
        self,
        centroids: Optional[torch.Tensor] = None,
        local_steering_matrices: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        strength: Optional[list] = None,
    ):
        device = next(self.parameters()).device
        if centroids is not None:
            centroids = centroids.to(device)
        if local_steering_matrices is not None:
            local_steering_matrices = local_steering_matrices.to(device)
        self.model.set_steering_parameters(
            centroids=centroids,
            local_steering_matrices=local_steering_matrices,
            temperature=temperature,
            strength=strength,
            device=device,
        )
