from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_cross_entropy(logits, targets, weights=None):
    losses = -(targets * F.log_softmax(logits.float(), dim=-1)).sum(-1)
    if weights is None:
        return losses.mean()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


class OrientationPredictor(nn.Module):
    """Shared A/B utterance scorer and role-aware next-response predictor.

    The frozen Mistral encodes roles in context tokens. The scorer is trained on
    *all* labelled utterances, including assistant turns; it is therefore also
    used to score newly generated replies before nearest-response retrieval.
    """

    def __init__(self, feature_dim: int, hidden_size: int, prototypes, transition):
        super().__init__()
        prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
        transition = torch.as_tensor(transition, dtype=torch.float32)
        clusters, vector_dim = prototypes.shape
        if transition.shape != (clusters, clusters):
            raise ValueError("Transition/prototype shape mismatch")
        self.register_buffer("prototypes", prototypes)
        self.register_buffer("transition", transition)
        self.features = nn.Sequential(nn.Linear(feature_dim, hidden_size), nn.Tanh())
        self.utterance_head = nn.Linear(hidden_size, clusters)
        self.source_vector_head = nn.Linear(hidden_size, vector_dim)
        target_dim = hidden_size + clusters + vector_dim
        self.target_head = nn.Sequential(nn.Linear(target_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, clusters))
        self.gate = nn.Linear(target_dim, 1)

    def utterance(self, features):
        hidden = self.features(features.float())
        logits = self.utterance_head(hidden)
        probability = logits.softmax(-1)
        return {"logits": logits, "probability": probability,
                "vector": probability @ self.prototypes,
                "direct_vector": self.source_vector_head(hidden)}

    def forward(self, context_features, source_features):
        source = self.utterance(source_features)
        target_input = torch.cat([self.features(context_features.float()), source["probability"], source["direct_vector"]], -1)
        logits = self.target_head(target_input)
        prior = source["probability"] @ self.transition
        # H(p)/log(K); with one region the uncertainty is exactly zero.
        entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1, keepdim=True)
        if logits.shape[-1] > 1:
            entropy = entropy / torch.log(torch.tensor(float(logits.shape[-1]), device=logits.device))
        else:
            entropy = torch.zeros_like(entropy)
        gate = torch.sigmoid(self.gate(target_input)) * entropy
        controlled_logits = logits + 0.5 * gate * prior.clamp_min(1e-8).log()
        probability = controlled_logits.softmax(-1)
        return {"logits": controlled_logits, "prior_free_logits": logits,
                "probability": probability, "vector": probability @ self.prototypes,
                "source": source}


class PrefixProjector(nn.Module):
    def __init__(self, vector_dim: int, hidden_size: int, lm_hidden_size: int, prefix_tokens: int = 8):
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.lm_hidden_size = lm_hidden_size
        self.project = nn.Sequential(nn.Linear(vector_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, prefix_tokens * lm_hidden_size))

    def forward(self, vectors):
        return self.project(vectors.float()).view(-1, self.prefix_tokens, self.lm_hidden_size)


def reply_loss(model, projector, vector, prompt_ids, reply_ids):
    """Causal CE only on reply + EOS; preserve gradients through frozen LM."""
    if prompt_ids.ndim != 2 or reply_ids.ndim != 2 or prompt_ids.shape[0] != 1 or reply_ids.shape[0] != 1:
        raise ValueError("Prefix training expects a single unpadded example")
    if reply_ids.shape[1] == 0:
        raise ValueError("An empty target cannot train a generator")
    ids = torch.cat([prompt_ids, reply_ids], dim=1)
    token_embeds = model.get_input_embeddings()(ids).detach()
    prefix = projector(vector).to(token_embeds.dtype)
    embeds = torch.cat([prefix, token_embeds], dim=1)
    labels = torch.full((1, embeds.shape[1]), -100, dtype=torch.long, device=ids.device)
    labels[:, prefix.shape[1] + prompt_ids.shape[1]:] = reply_ids
    attention_mask = torch.ones(labels.shape, dtype=torch.long, device=ids.device)
    return model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels, use_cache=False).loss
