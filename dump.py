import os
import math
import torch
import pickle
import forgetting_transformer.model  # Needed to register the model classes
import forgetting_transformer.tokenizer  # Needed to register the tokenizer class
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from einops import rearrange
from types import MethodType
from typing import List, Optional, Tuple, Union
from fla.modules.activations import swiglu_linear
from transformers.cache_utils import Cache
from forgetting_transformer.model.forgetting_transformer.modeling_forgetting_transformer import ForgettingAttentionLayer
from forgetting_transformer.ops.forgetting_attention import forgetting_attention
from forgetting_transformer.model.forgetting_transformer.fgate_cache import FgateDynamicCache
from forgetting_transformer.model.forgetting_transformer.glu_linear import glu_linear
from forgetting_transformer.model.forgetting_transformer.token_shift import token_shift

data_tmp_path = "fox-wikitext.bin"

def get_data(tok):
    num_samples = 4
    max_length = 8192
    if os.path.exists(data_tmp_path):
        return torch.load(data_tmp_path)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    cur = 0
    ret = []
    sample = []
    while len(ret) < num_samples:
        sample.append(ds[cur]['text'])
        cur += 1
        tokenized = tok("".join(sample))['input_ids']
        if len(tokenized) >= max_length:
            ret.append(
                torch.tensor(tokenized)[:max_length]
            )
            sample = []
    ret = torch.stack(ret)
    torch.save(ret, data_tmp_path)
    return ret

def dump_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        We assume that during decoding attention mask is always 1. Otherwise it won't work.
        """
        batch_size, q_len, _ = hidden_states.size()
        if use_cache:
            key_shift_state = past_key_values.key_shift_cache[self.layer_idx]
            value_shift_state = past_key_values.value_shift_cache[self.layer_idx]
        else:
            key_shift_state = value_shift_state = None

        # Shift states are updated in place
        q = self.q_proj(hidden_states)
        if self.use_k_shift:
            k = self.k_proj(hidden_states, key_shift_state)
        else:
            k = self.k_proj(hidden_states)
        if self.use_v_shift:
            v = self.v_proj(hidden_states, value_shift_state)
        else:
            v = self.v_proj(hidden_states)

        if self.qk_norm and (not self.qk_norm_share_param_across_head):
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)

        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        k = rearrange(k, '... (h d) -> ... h d', h=self.num_kv_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_kv_heads)


        if self.qk_norm and (self.qk_norm_share_param_across_head):
            q = self.q_norm(q).to(q.dtype)
            k = self.k_norm(k).to(k.dtype)


        seqlen_offset, max_seqlen = 0, q.shape[1]
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = (seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1])
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        if self.rotary is not None:
            q, k = self.rotary(q, k, seqlen_offset, max_seqlen)

        if self.fgate_type == "full":
            fgate_logit = self.fgate_proj(hidden_states)
            fgate_logit = rearrange(fgate_logit, "b t h -> b h t")
            log_fgate = torch.nn.functional.logsigmoid(fgate_logit.float())
        elif self.fgate_type == "none":
            log_fgate = torch.zeros((batch_size, self.num_heads, q_len), dtype=torch.float32, device=hidden_states.device)
        else:
            assert self.fgate_type in ["fixed", "bias_only"]
            fgate_logit = torch.broadcast_to(self.fgate_bias, (batch_size, q_len, self.num_heads))
            fgate_logit = rearrange(fgate_logit, "b t h -> b h t")
            log_fgate = torch.nn.functional.logsigmoid(fgate_logit.float())

        k = rearrange(k, 'b t h d -> b h t d')
        if past_key_values is not None:
            k, v, log_fgate = past_key_values.update(k, v, log_fgate, self.layer_idx)
        # k, v = rearrange(k, 'b h t d -> b t h d'), rearrange(v, 'b h t d -> b t h d')
        q = rearrange(q, 'b t h d -> b h t d')

        if self.num_kv_groups > 1:
            assert False
            k = rearrange(k.unsqueeze(-2).repeat(1, 1, 1, self.num_kv_groups, 1), 'b t h g d -> b t (h g) d')
            v = rearrange(v.unsqueeze(-2).repeat(1, 1, 1, self.num_kv_groups, 1), 'b t h g d -> b t (h g) d')
        # with open(f"tensor/{self.layer_idx}_q.pkl", "wb") as f:
        #     pickle.dump(q, f)
        # with open(f"tensor/{self.layer_idx}_k.pkl", "wb") as f:
        #     pickle.dump(k, f)
        # with open(f"tensor/{self.layer_idx}_v.pkl", "wb") as f:
        #     pickle.dump(v, f)
        # with open(f"tensor/{self.layer_idx}_gk.pkl", "wb") as f:
        #     pickle.dump(log_fgate, f)
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            breakpoint()
            B, _, T = log_fgate.size()
            assert attention_mask.size() == (B, T), ((B, T), attention_mask.size())
            seq_start = T - attention_mask.sum(dim=-1)
            o = forgetting_attention(
                q, k, v,
                log_fgate,
                head_first=True,
                seq_start=seq_start,
                sm_scale=1 / math.sqrt(self.head_dim),
            )
            o = rearrange(o, "b h t d -> b t h d")
        else:
            o = forgetting_attention(
                q, k, v,
                log_fgate,
                head_first=True,
                sm_scale=1 / math.sqrt(self.head_dim),
            )
            o = rearrange(o, "b h t d -> b t h d")

        o = o.reshape(batch_size, q_len, self.hidden_size)

        if self.output_norm is not None:
            o = self.output_norm(o)

        if self.ogate_proj is not None:
            # ogate = self.ogate act(self.ogate_proj(hidden_states))
            # o = o * ogate
            # ogate = act_gate(self.ogate_proj(hidden_states), o)
            ogate_logit = self.ogate_proj(hidden_states)
            dtype = ogate_logit.dtype
            if self.ogate_act == "silu":
                o = swiglu_linear(ogate_logit, o, self.o_proj.weight.to(dtype), self.o_proj.bias.to(dtype) if self.o_proj.bias is not None else self.o_proj.bias)
            elif self.ogate_act == "sigmoid":
                o = glu_linear(ogate_logit, o, self.o_proj.weight.to(dtype), self.o_proj.bias.to(dtype) if self.o_proj.bias is not None else self.o_proj.bias)
            else:
                raise ValueError(f"Unknown ogate act {self.ogate_act}")
        else:
            o = self.o_proj(o)

        if not output_attentions:
            attentions = None
        else:
            SAVE_HEADS = [0, 1, 2, 3]
            # (B, H, T, T)
            score = q[:, SAVE_HEADS] @ k[:, SAVE_HEADS].mT
            log_lambda = torch.cumsum(log_fgate, dim=-1)
            decay_bias = (log_lambda[:, SAVE_HEADS, :, None] - log_lambda[:, SAVE_HEADS, None, :]).to(torch.bfloat16)
            # normalized_score = torch.softmax(score, dim=-1)
            attentions = (score, decay_bias)

        return o, attentions, past_key_values

def set_attn_fox(model):
    for layer_id, layer in enumerate(model.model.layers):
        layer.attn.forward = MethodType(dump_forward, layer.attn)


model = AutoModelForCausalLM.from_pretrained("/ssd/data/weijia/flash-linear-attention/tmp-data/model/fla-hub/fox-pro-760m-longcrawl64-48b")
tokenizer = AutoTokenizer.from_pretrained("/ssd/data/weijia/flash-linear-attention/tmp-data/model/fla-hub/fox-pro-760m-longcrawl64-48b", add_bos_token=True, clean_up_tokenization_spaces=False)
set_attn_fox(model)
data = get_data(tok = tokenizer).cuda()
model = model.cuda()

with torch.no_grad():
    output = model(data)