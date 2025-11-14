from dataclasses import dataclass
import pdb 
import math
from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5Block,
    T5Attention,
    T5LayerNorm,
    T5LayerSelfAttention,
    T5LayerFF,
    T5LayerCrossAttention,
    T5PreTrainedModel,
    T5ForConditionalGeneration,
)

from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy

from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer, T5Tokenizer, T5Config

import argparse
import sys 
from utils.generation import Explore_GenerationMixin

logger = logging.get_logger(__name__)

class UnusedEncoder(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False


    def forward(
        self,
        backbone = None,
        encoder_attention_mask=None,
        hidden_states=None,
        decoder_hidden_states=None,
        lm_logits=None,
        decoder_attention_mask=None, # None
        past_key_values=None, # None 
        use_cache=None, # None 
        labels=None,
        decoder_inputs_embeds=None, # None
        head_mask=None, # None
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_attention=None,
        reduce_loss=False,
        loss_type =None,
        **kwargs,
    ):
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=output_attentions,
            cross_attentions=output_attentions,
        )

class JointDecoder(T5Stack): # Exploration Network
    def __init__(self,config,embed_tokens=None):
        super(T5Stack, self).__init__(config)
        
        self.config = config
        self.d_model = self.config.d_model
        
        self.embed_tokens = embed_tokens
        self.is_decoder = self.config.is_decoder
        assert self.config.is_decoder is True 
        
        #Fully connected layer to process the gradient tensor array
        self.fc_layer = nn.Linear(32100, self.config.d_model)
        
        ## Uncomment if using expanded forward
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        explore_hidden_states=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        explore_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            # input_ids = input_ids.view(-1, input_shape[-1])
            input_shape = input_shape[:-1]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.fc_layer(input_ids)
            

        batch_size, seq_length = input_shape

        encoder_hidden_states = torch.cat((encoder_hidden_states,explore_hidden_states), dim=1) # [64,189,512]
        encoder_attention_mask = torch.cat((encoder_attention_mask,explore_attention_mask), dim=1) # [64,189]
        
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

class exploration_net(T5ForConditionalGeneration, Explore_GenerationMixin):
    
    def __init__(self, config):
        super(T5ForConditionalGeneration, self).__init__(config)
        
        self.config = config
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        # NOTE: encoder is not used in the exploration network
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UnusedEncoder(encoder_config, self.shared)
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.use_cache = False
        decoder_config.is_encoder_decoder = False 
        self.decoder = JointDecoder(decoder_config,self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.init_weights()
        
        self.explore_weight = 0
        self.model_parallel = False
        self.device_map = None
        self.sigmoid = nn.Sigmoid()
        self.grad_lists = []
        self.prob_lists = []
        self.encoder_hidden_states_lists = []
        self.explore_logits = None 
    
    def forward(
        self,
        backbone = None,
        encoder_attention_mask=None,
        hidden_states=None,
        decoder_hidden_states=None,
        lm_logits=None,
        decoder_attention_mask=None, 
        past_key_values=None, 
        use_cache=None, 
        labels=None,
        decoder_inputs_embeds=None, 
        head_mask=None, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_attention=None,
        reduce_loss=False,
        loss_type =None,
        decoder_input_ids=None,
        encoder_outputs = None,
        **kwargs,
    ):
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        if labels is not None and decoder_input_ids is None: # training
            # Calculate probabilities
            probabilities = F.softmax(lm_logits, dim=-1)
            explore_labels_prob = torch.zeros_like(probabilities,device=self.device,dtype=hidden_states.dtype)
            explore_labels_prob[
                torch.arange(labels.size()[0], device=self.device).unsqueeze(1).expand(-1, labels.size()[1]), 
                torch.arange(labels.size()[1], device=self.device).unsqueeze(0).expand(labels.size()[0], -1), 
                labels
            ] = torch.tensor(1, device=self.device,dtype=hidden_states.dtype)
            explore_labels_prob = explore_labels_prob - probabilities
            

        decoder_hidden_states_l2norm = torch.norm(decoder_hidden_states, p=2, dim=2, keepdim=False)
        explore_attention_mask = decoder_hidden_states_l2norm.ne(self.config.pad_token_id).to(
            dtype=torch.long, device=decoder_hidden_states.device
        )


        if decoder_input_ids is None: # training
            explore_outputs = self.decoder(
                input_ids=explore_labels_prob, 
                attention_mask=decoder_attention_mask, 
                inputs_embeds=decoder_inputs_embeds, 
                past_key_values=past_key_values, 
                explore_hidden_states=decoder_hidden_states, 
                encoder_hidden_states=hidden_states, 
                encoder_attention_mask=encoder_attention_mask, 
                explore_attention_mask=explore_attention_mask, 
                head_mask=head_mask, 
                use_cache=use_cache, 
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states, 
                return_dict=return_dict,
            )
        
        else: # generation
            explore_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask, 
                inputs_embeds=decoder_inputs_embeds, 
                past_key_values=past_key_values, 
                explore_hidden_states=decoder_hidden_states,
                encoder_hidden_states=hidden_states, 
                encoder_attention_mask=encoder_attention_mask,
                explore_attention_mask=explore_attention_mask, 
                head_mask=head_mask, 
                use_cache=use_cache, 
                output_attentions=output_attentions, 
                output_hidden_states=output_hidden_states, 
                return_dict=return_dict,
            )

        explore_sequence_output = explore_outputs[0]
        explore_lm_logits = self.lm_head(explore_sequence_output)
        explore_lm_logits = explore_lm_logits / 100
        
        explore_loss = None
        if labels is not None: # training
            if loss_type == "ce":
                if explore_labels_prob is not None:
                    if reduce_loss:
                        loss_fct = CrossEntropyLoss(ignore_index=-100)
                    else:
                        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
                explore_probabilities = F.softmax(explore_lm_logits, dim=-1)
                explore_loss = loss_fct(explore_probabilities.view(-1, explore_probabilities.size(-1)), labels.view(-1)) 
                
            
            elif loss_type == "rmse":
                epsilon = 1e-8  # Small value to prevent numerical instability
                mse_loss = torch.mean((explore_lm_logits - explore_labels_prob) ** 2)
                explore_loss = torch.sqrt(mse_loss + epsilon)

        else: # generation
            self.explore_logits = explore_lm_logits

            
        return P5Seq2SeqLMOutput(
            explore_loss=explore_loss,
            # logits=sum_logits if labels is None else explore_lm_logits, 
            logits=explore_lm_logits,
            past_key_values=explore_outputs.past_key_values,
            decoder_last_hidden_state=explore_outputs.last_hidden_state,
            decoder_hidden_states=explore_outputs.hidden_states,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        hidden_states=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        lm_logits = None,
        loss_type = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        decoder_hidden_states=None,
        **kwargs
    ):
        
        batch_size, seq_len, vocab_size = lm_logits.size()
        if past_key_values is not None: 
            pad_prob = torch.zeros(batch_size, 1, vocab_size).to(lm_logits.device) 
            explore_prob = torch.cat((pad_prob,self.explore_logits),dim=1)
        else:
            explore_prob = torch.zeros(batch_size, 1, vocab_size).to(lm_logits.device)  # Zero vector for initialization
        
        
        return {
            "decoder_input_ids": explore_prob,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "lm_logits": lm_logits,
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_hidden_states": decoder_hidden_states,    
            "loss_type": loss_type,
        }

        
@dataclass
class P5Seq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see ``past_key_values`` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    explore_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attention_mask: Optional[torch.FloatTensor] = None


@dataclass
class P5AdversarialSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see ``past_key_values`` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    rec_loss: Optional[torch.FloatTensor] = None
    discriminator_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    feature_embeddings: Optional[torch.FloatTensor] = None


if __name__ == "__main__":
    config = T5Config.from_pretrained('t5-small')
    model = exploration_net.from_pretrained('t5-small', config=config)
    model.to('cuda')
    
