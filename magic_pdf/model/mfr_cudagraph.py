from typing import Optional, Tuple, Union
import torch
from torch import nn
import os
from unimernet.common.config import Config
import unimernet.tasks as tasks
import argparse
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask

class PatchedMBartLearnedPositionalEmbedding(nn.Module):

    def __init__(self, origin: nn.Module):
        super().__init__()
        self.offset = origin.offset
        self.embedding = nn.Embedding(origin.num_embeddings, origin.embedding_dim)
        self.embedding.weight.data = origin.weight.data

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.embedding.weight.device
        )
        positions += past_key_values_length
        positions = positions.expand(bsz, -1)

        return self.embedding(positions + self.offset)


class PatchedMBartDecoder(nn.Module):
    def __init__(self, origin: nn.Module, kvlen: torch.LongTensor):
        super().__init__()
        self.origin = origin
        self.kvlen = kvlen

        self.config = origin.config
        self.embed_tokens = origin.embed_tokens
        self.embed_scale = origin.embed_scale
        self._use_flash_attention_2 = origin._use_flash_attention_2
        self.embed_positions = origin.embed_positions
        self.counting_context_weight = getattr(origin, 'counting_context_weight', None)
        self.layernorm_embedding = origin.layernorm_embedding
        self.layers = origin.layers
        self.layer_norm = origin.layer_norm

        self.patched_embed_positions = PatchedMBartLearnedPositionalEmbedding(self.embed_positions)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        count_pred: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        run_origin = False
        if past_key_values is None:
            run_origin = True
        elif past_key_values[0][0].size(-2) < attention_mask.size(-1):
            run_origin = True

        if run_origin:
            return self.origin(
                input_ids=input_ids,
                attention_mask=attention_mask,
                count_pred=count_pred,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # embed positions
        positions = self.patched_embed_positions(input, self.kvlen)

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)

        # TODO: add counting context weight to hidden_states
        if count_pred is not None:
            count_context_weight = self.counting_context_weight(count_pred)
            hidden_states = hidden_states + 0.5 * count_context_weight.unsqueeze(1)
        hidden_states = self.layernorm_embedding(hidden_states)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class PatchedMBartAttention(nn.Module):

    def __init__(self, origin: nn.Module, kvlen: torch.LongTensor):
        super().__init__()
        self.embed_dim = origin.embed_dim
        self.num_heads = origin.num_heads
        self.dropout = origin.dropout
        self.head_dim = origin.head_dim
        self.config = origin.config

        self.scaling = origin.scaling
        self.is_decoder = origin.is_decoder
        self.is_causal = origin.is_causal

        self.k_proj = origin.k_proj
        self.v_proj = origin.v_proj
        self.q_proj = origin.q_proj
        self.out_proj = origin.out_proj
        self.kvlen = kvlen

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

            if past_key_value[0].size(-2) < attention_mask.size(-1):
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            else:
                past_key_value[0][:, :, self.kvlen[None]] = key_states
                past_key_value[1][:, :, self.kvlen[None]] = value_states
                key_states = past_key_value[0]
                value_states = past_key_value[1]
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = attn_weights

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        # attn_output = self.out_proj(attn_output)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class PatchedMBartSqueezeAttention(nn.Module):

    def __init__(self, origin: nn.Module, kvlen: torch.LongTensor):
        super().__init__()
        self.embed_dim = origin.embed_dim
        self.num_heads = origin.num_heads
        self.dropout = origin.dropout
        self.head_dim = origin.head_dim
        self.squeeze_head_dim=origin.squeeze_head_dim
        self.config = origin.config

        self.scaling = origin.scaling
        self.is_decoder = origin.is_decoder
        self.scaling = origin.scaling

        self.q_proj = origin.q_proj
        self.k_proj = origin.k_proj
        self.v_proj = origin.v_proj
        self.out_proj = origin.out_proj
        self.kvlen = kvlen

    def _shape_qk(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.squeeze_head_dim).transpose(1, 2).contiguous()

    def _shape_v(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape_qk(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape_v(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape_qk(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape_v(self.v_proj(hidden_states), -1, bsz)

            if past_key_value[0].size(-2) < attention_mask.size(-1):
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            else:
                past_key_value[0][:, :, self.kvlen[None]] = key_states
                past_key_value[1][:, :, self.kvlen[None]] = value_states
                key_states = past_key_value[0]
                value_states = past_key_value[1]
        else:
            # self_attention
            key_states = self._shape_qk(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape_v(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.squeeze_head_dim)
        value_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape_qk(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*value_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

def patch_model(model: nn.Module, kvlen: torch.LongTensor):
    for name, child in model.named_children():
        cls_name = type(child).__name__
        if cls_name == 'MBartAttention':
            patched_child = PatchedMBartAttention(child, kvlen)
            model.register_module(name, patched_child)
        elif cls_name == 'MBartSqueezeAttention':
            patched_child = PatchedMBartSqueezeAttention(child, kvlen)
            model.register_module(name, patched_child)
        else:
            patch_model(child, kvlen)

    cls_name = type(model).__name__
    if cls_name == 'CustomMBartDecoder':
        model = PatchedMBartDecoder(model, kvlen)
    return model


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def get_graph_key(batch_size: int, kvlens: int):
    batch_size = next_power_of_2(batch_size)
    kvlens = next_power_of_2(kvlens)

    batch_size = max(8, batch_size)
    kvlens = max(32, kvlens)

    return batch_size, kvlens


class GraphRunnerImpl:

    def __init__(self, model: nn.Module, graph: torch.cuda.CUDAGraph, input_buffers: dict, output_buffers: dict):
        self.model = model
        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = output_buffers

    @staticmethod
    def extract_input_buffers(input_buffers: dict, batch_size: int, kvlens: int):
        input_ids = input_buffers['input_ids'][:batch_size]
        attention_mask = input_buffers['attention_mask'][:batch_size, :kvlens]
        encoder_hidden_states = input_buffers['encoder_hidden_states'][:batch_size]
        kvlen=input_buffers['kvlen']

        past_key_values = []
        for past_key_value in input_buffers['past_key_values']:
            k0 = past_key_value[0][:batch_size, :, :kvlens]
            v0 = past_key_value[1][:batch_size, :, :kvlens]
            k1 = past_key_value[2][:batch_size]
            v1 = past_key_value[3][:batch_size]
            past_key_values.append((k0, v0, k1, v1))

        input_buffers = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            kvlen=kvlen,
        )
        return input_buffers

    @staticmethod
    def fill_input_buffers(
        input_buffer: dict,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        ):
        batch_size = input_ids.size(0)
        kvlens = attention_mask.size(1)

        input_buffer['input_ids'][:batch_size] = input_ids

        if input_buffer['attention_mask'].data_ptr() != attention_mask.data_ptr():
            input_buffer['attention_mask'].fill_(0)
        input_buffer['attention_mask'][:batch_size, :kvlens] = attention_mask
        input_buffer['encoder_hidden_states'][:batch_size] = encoder_hidden_states

        if past_key_values is not None:
            for buf_kv, kv in zip(input_buffer['past_key_values'], past_key_values):
                idx = 0
                if buf_kv[idx].data_ptr() != kv[idx].data_ptr():
                    buf_kv[idx].fill_(0)
                    buf_kv[idx][:batch_size, :, :kvlens-1] = kv[idx]
                idx = 1
                if buf_kv[idx].data_ptr() != kv[idx].data_ptr():
                    buf_kv[idx].fill_(0)
                    buf_kv[idx][:batch_size, :, :kvlens-1] = kv[idx]

                idx = 2
                if buf_kv[idx].data_ptr() != kv[idx].data_ptr():
                    buf_kv[idx].fill_(0)
                    buf_kv[idx][:batch_size] = kv[idx]
                idx = 3
                if buf_kv[idx].data_ptr() != kv[idx].data_ptr():
                    buf_kv[idx].fill_(0)
                    buf_kv[idx][:batch_size] = kv[idx]
        
        input_buffer['kvlen'].fill_(kvlens - 1)
    
    @classmethod
    @torch.inference_mode()
    def capture(cls,
                model: nn.Module,
                input_buffers: dict,
                pool,
                warmup: bool = False,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                count_pred: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
        batch_size = input_ids.size(0)
        kvlens = attention_mask.size(1)

        graph_key = get_graph_key(batch_size, kvlens)
        batch_size = graph_key[0]
        kvlens = graph_key[1]

        input_buffers = cls.extract_input_buffers(input_buffers,
                                                  batch_size=batch_size,
                                                  kvlens=kvlens)
        cls.fill_input_buffers(input_buffers,
                               input_ids,
                               attention_mask,
                               encoder_hidden_states,
                               past_key_values)
        
        input_ids = input_buffers['input_ids']
        attention_mask = input_buffers['attention_mask']
        encoder_hidden_states = input_buffers['encoder_hidden_states']
        past_key_values = input_buffers['past_key_values']

        if warmup:
            # warmup
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                count_pred=count_pred,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,
                              pool=pool):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                count_pred=count_pred,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)

        output_buffers = dict(
            last_hidden_state=outputs['last_hidden_state'],
            past_key_values=outputs['past_key_values'],
            )

        return GraphRunnerImpl(model, graph, input_buffers, output_buffers)

    def __call__(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        count_pred: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        batch_size = input_ids.size(0)
        kvlens = attention_mask.size(1)
        self.fill_input_buffers(self.input_buffers,
                               input_ids,
                               attention_mask,
                               encoder_hidden_states,
                               past_key_values)

        self.graph.replay()

        last_hidden_state = self.output_buffers['last_hidden_state'][:batch_size]
        
        past_key_values = []
        for past_key_value in self.output_buffers['past_key_values']:
            k0 = past_key_value[0][:batch_size, :, :kvlens]
            v0 = past_key_value[1][:batch_size, :, :kvlens]
            k1 = past_key_value[2][:batch_size]
            v1 = past_key_value[3][:batch_size]
            past_key_values.append((k0, v0, k1, v1))
        
        outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
        )
        return outputs

class GraphRunner(nn.Module):

    def __init__(self, model: nn.Module, max_batchs: int, max_kvlens: int, dtype:torch.dtype = torch.float16, device: torch.device = 'cuda'):
        super().__init__()

        self.kvlen = torch.tensor(0, dtype=torch.long, device=device)
        model = patch_model(model.to(dtype), self.kvlen)
        self.model = model
        self.max_batchs = max_batchs
        self.max_kvlens = max_kvlens
        self.device = device

        self.input_buffers = None

        self.impl_map = dict()
        self.graph_pool_handle = torch.cuda.graph_pool_handle()
        self.warmuped = False

    def create_buffers(self, encoder_kvlens: int, dtype: torch.dtype):
        max_batchs = self.max_batchs
        max_kvlens = self.max_kvlens
        device = self.device
        config = self.model.config

        d_model = config.d_model
        decoder_layers = config.decoder_layers
        num_heads = config.decoder_attention_heads

        head_dim = d_model // num_heads
        self_attn = self.model.layers[0].self_attn
        qk_head_dim = getattr(self_attn, 'squeeze_head_dim', head_dim)

        input_ids = torch.ones((max_batchs, 1), dtype=torch.int64, device=device)
        attention_mask = torch.zeros((max_batchs, max_kvlens), dtype=torch.int64, device=device)
        encoder_hidden_states = torch.zeros((max_batchs, encoder_kvlens, d_model), dtype=dtype, device=device)

        past_key_values = []
        for _ in range(decoder_layers):
            k0 = torch.zeros((max_batchs, num_heads, max_kvlens, qk_head_dim), dtype=dtype, device=device)
            v0 = torch.zeros((max_batchs, num_heads, max_kvlens, head_dim), dtype=dtype, device=device)
            k1 = torch.zeros((max_batchs, num_heads, encoder_kvlens, qk_head_dim), dtype=dtype, device=device)
            v1 = torch.zeros((max_batchs, num_heads, encoder_kvlens, head_dim), dtype=dtype, device=device)

            past_key_values.append((k0, v0, k1, v1))

        self.input_buffers = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            kvlen=self.kvlen
        )

    @torch.inference_mode()
    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        count_pred: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        batch_size, qlens = input_ids.size()
        kvlens = attention_mask.size(1)

        eager_mode = False

        if qlens != 1:
            eager_mode = True

        if past_key_values is None:
            eager_mode = True
        else:
            for past_key_value in past_key_values:
                if past_key_value is None:
                    eager_mode = True
                    break

        if batch_size >= self.max_batchs or kvlens >= self.max_kvlens:
            eager_mode = True

        if eager_mode:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                count_pred=count_pred,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,)
        
        # create buffer if not exists.
        if self.input_buffers is None:
            encoder_kvlens = encoder_hidden_states.size(1)
            self.create_buffers(encoder_kvlens=encoder_kvlens, dtype=encoder_hidden_states.dtype)

        graph_key = get_graph_key(batch_size, kvlens)
        if graph_key not in self.impl_map:
            warmup = False
            if not self.warmuped:
                warmup = True
                self.warmuped = True
            impl = GraphRunnerImpl.capture(
                self.model,
                self.input_buffers,
                self.graph_pool_handle,
                warmup=warmup,
                input_ids=input_ids,
                attention_mask=attention_mask,
                count_pred=count_pred,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            self.impl_map[graph_key] = impl
        impl = self.impl_map[graph_key]

        ret = impl(
                input_ids=input_ids,
                attention_mask=attention_mask,
                count_pred=count_pred,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )
        return ret
