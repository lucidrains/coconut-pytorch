from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, cat, stack, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.checkpoint import checkpoint_sequential

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def min_p_filter(logits, min_p = 0.1):
    """
    https://arxiv.org/abs/2407.01082
    """
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# swiglu feedforward, Shazeer et al.

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        rotary_pos_emb: RotaryEmbedding | None = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.rotary_pos_emb = rotary_pos_emb

        self.norm = nn.RMSNorm(dim)

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        cached_kv: Tensor | None = None,
        return_cached_kv = False
    ):

        x = self.norm(x)

        qkv = self.to_qkv(x)

        q, k, v = self.split_heads(qkv)

        if exists(cached_kv):
            ck, cv = cached_kv

            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

        if return_cached_kv:
            cached_kv = stack((k, v))

        if exists(self.rotary_pos_emb):
            q, k = self.rotary_pos_emb.rotate_queries_with_cached_keys(q, k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True
        )

        out = self.merge_heads(out)

        out = self.to_out(out)

        if not return_cached_kv:
            return out

        return out, cached_kv

# main class

class Transformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.rotary_emb = RotaryEmbedding(dim_head)

        layers = ModuleList([])

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, rotary_pos_emb = self.rotary_emb),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.layers = layers

        self.norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        inp: Tensor | list[Tensor],
        cached_kv: Tensor | None = None,
        return_intermediates = False,
        return_embed_with_cache_kv = False
    ):

        # handle input, which can be a list of Tensor of Float['b n d'] or Int['b n']

        if not isinstance(inp, list):
            inp = [inp]

        inp = [self.token_emb(t) if t.dtype in (torch.int, torch.long) else t for t in inp]

        x = cat(inp, dim = -2)

        # cached key values need to be handled with priority and care for this paper

        cached_kv = default(cached_kv, [])
        cached_kv_iter = iter(cached_kv)

        next_keys_values = []

        for attn, ff in self.layers:

            attn_out, key_values = attn(
                x,
                cached_kv = next(cached_kv_iter, None),
                return_cached_kv = True
            )

            x = attn_out + x

            next_keys_values.append(key_values)

            x = ff(x) + x

        embeds = self.norm(x)

        if return_embed_with_cache_kv:
            return embeds, next_keys_values

        logits = self.to_logits(embeds)

        if not return_intermediates:
            return logits

        return logits, embeds, next_keys_values

# coconut wrapper around transformer handles recurrence with latent reasoning tokens

class Coconut(Module):
    def __init__(
        self,
        num_reasoning_steps,
        transformer: dict | Transformer,
        num_latents_per_step = 1, # extending the paper, allow for more than one "reasoning" token per step
        learn_begin_of_thought = False,
        checkpoint = False
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(**transformer)

        dim = transformer.dim

        self.model = transformer
        self.num_reasoning_steps = num_reasoning_steps

        # begin and end of thought tokens, handled external to transformer

        self.begin_of_thought = nn.Parameter(torch.zeros(num_latents_per_step, dim))
        self.end_of_thought = nn.Parameter(torch.zeros(dim))

        nn.init.normal_(self.begin_of_thought, std = 0.02)
        nn.init.normal_(self.end_of_thought, std = 0.02)

        # whether to teach model when to begin a thought

        self.learn_begin_of_thought = learn_begin_of_thought
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # checkpoint

        self.checkpoint = checkpoint

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_length = 16,
        filter_fn = min_p_filter,
        filter_kwargs: dict = dict(),
        temperature = 1.
    ):
        prompt_logits, latent_tokens, answer_logits, cached_kv = self.forward(prompt)

        out = prompt[:, 0:0]

        def sample(logits):
            nonlocal out
            logits = filter_fn(logits[:, -1], **filter_kwargs)
            sampled = gumbel_sample(logits, temperature = temperature)
            out = cat((out, sampled), dim = -1)

        sample(answer_logits)

        for _ in range(max_length - 1):
            answer_logits = self.model(out[:, -1:], cached_kv = cached_kv)
            sample(answer_logits)

        return out

    def checkpointed_recurrent_latent_forward(
        self,
        latent_token,
        cached_kv
    ):
        if not torch.is_tensor(cached_kv):
            cached_kv = stack(cached_kv)

        num_recurrent_steps = self.num_reasoning_steps - 1

        # recurrent model forward with next latent token + past cached kv, but checkpointed

        latent_tokens = [latent_token, *((None,) * num_recurrent_steps)]

        def recurrent_step(step_inputs):
            i, latent_token, *latent_tokens, cached_kv = step_inputs

            i += 1

            latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, return_embed_with_cache_kv = True)

            latent_tokens[i] = latent_token

            return (i, latent_token, *latent_tokens, stack(cached_kv))

        # functions

        fns = [recurrent_step] * num_recurrent_steps

        # initial input

        inputs = (0, latent_token, *latent_tokens, cached_kv)

        # forward checkpoint sequential

        _, latent_token, *latent_tokens, cached_kv = checkpoint_sequential(fns, 1, input = inputs, use_reentrant = False)

        return latent_token, latent_tokens, cached_kv

    def recurrent_latent_forward(
        self,
        latent_token,
        cached_kv
    ):
        # latent reasoning is a recurrent model forward with the last hidden state being passed back in as input, while the prompt key / values are kept the same (prompt is NOT passed back in)

        latent_tokens = [latent_token]

        for _ in range(self.num_reasoning_steps - 1):
            latent_token, cached_kv = self.model(latent_token, cached_kv = cached_kv, return_embed_with_cache_kv = True)

            latent_tokens.append(latent_token)

        return latent_token, latent_tokens, cached_kv

    def forward(
        self,
        prompt,
        answer = None,
        return_loss = True,
        return_intermediates = False
    ):
        """
        ein notation:
        b - batch
        h - attention heads
        n - seq (number of reasoning tokens, etc)
        d - feature dimension
        l - logits (num tokens)
        """

        batch, num_thoughts = prompt.shape[0], self.begin_of_thought.shape[0]

        # prepare <bot> and <eot> in paper

        begin_thought = repeat(self.begin_of_thought, 'n d -> b n d', b = batch)
        end_thought = repeat(self.end_of_thought, 'd -> b 1 d', b = batch)

        # give the model the prompt

        prompt_logits, embeds, cached_kv = self.model([prompt, begin_thought], return_intermediates = True)

        # loss for decoding a <bot>

        bot_loss = self.zero

        if self.learn_begin_of_thought:
            pred_bot_embed, rest_logits = embeds[:, -2], prompt_logits[:, -2]

            pred_bot_logits = einsum('b d, n d -> b', pred_bot_embed, self.begin_of_thought)
            pred_bot_logits = rearrange(pred_bot_logits, 'b -> b 1')

            bot_logits = cat((pred_bot_logits, rest_logits), dim = -1)
            bot_labels = bot_logits.new_zeros((batch,), dtype = torch.long)

            bot_loss = F.cross_entropy(bot_logits, bot_labels)

        # extract latent reasoning token off <bot> position

        latent_token = embeds[:, -num_thoughts:]

        # whether to checkpoint or not

        should_checkpoint = self.training and self.checkpoint and latent_token.requires_grad

        # route to right functions

        if should_checkpoint:
            latent_token, latent_tokens, cached_kv = self.checkpointed_recurrent_latent_forward(latent_token, cached_kv)
        else:
            latent_token, latent_tokens, cached_kv = self.recurrent_latent_forward(latent_token, cached_kv)

        # final model forward inputs

        final_forward = [latent_token, end_thought]

        if exists(answer):
            final_forward.append(answer[..., :-1])

        # final step, latent token and end thought token, as well as answer sequence is appended together

        logits = self.model(final_forward, cached_kv = cached_kv)

        answer_logits = logits[:, num_thoughts:]

        # concat the latent reasoning tokens to be passed out for study

        latent_tokens = cat(latent_tokens, dim = -2)

        intermediates = prompt_logits, latent_tokens, answer_logits, cached_kv

        if not return_loss or not exists(answer):
            return intermediates

        # handle the loss on the answers

        answer_loss = F.cross_entropy(
            rearrange(answer_logits, 'b n l -> b l n'),
            answer
        )

        loss_breakdown = (answer_loss, bot_loss)

        total_loss = (answer_loss + bot_loss)

        if not return_intermediates:
            return total_loss

        return total_loss, (loss_breakdown, *intermediates)
