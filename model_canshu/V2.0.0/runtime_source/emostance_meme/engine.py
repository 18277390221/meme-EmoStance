from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import torch

from .backbone import LocalMistral
from .config import work_path
from .networks import OrientationPredictor, PrefixProjector
from .retrieval import ResponseIndex
from .space import EmotionSpace


class MemeEngine:
    def __init__(self, run_dir="runs/main", *, device=None, dtype=None):
        root = work_path(run_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"找不到完成的训练产物：{manifest_path}。请先准备数据并运行 train。")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1 or manifest.get("status") != "complete" or manifest.get("index_split") != "train":
            raise ValueError("训练产物未完成，或检索库不是仅由训练集构建")
        self.config = manifest["config"]
        self.space = EmotionSpace.load(root / "space.json")
        self.index = ResponseIndex.load(root / "responses.json", metric=self.config["distance"])
        if self.index.dimension != self.space.vector_dim or self.space.vector_dim != manifest["vector_dim"] or self.space.num_clusters != manifest["num_clusters"]:
            raise ValueError("检索库、情绪空间和 manifest 的维度不一致")
        train_ids = set(manifest["train_dialogue_ids"])
        if any(str(row["dialogue_id"]) not in train_ids for row in self.index.records):
            raise ValueError("检索库混入非训练对话，拒绝加载")
        self.backbone = LocalMistral(self.config["base_model"], device=device or self.config["device"], dtype=dtype or self.config["dtype"], max_context_tokens=self.config["max_context_tokens"])
        if self.backbone.hidden_size != manifest["feature_dim"]:
            raise ValueError("当前基座特征维度与训练产物不一致")
        self.predictor = OrientationPredictor(manifest["feature_dim"], self.config["head_hidden_size"], self.space.prototypes, np.eye(self.space.num_clusters)).to(self.backbone.device)
        self.projector = PrefixProjector(self.space.vector_dim, self.config["projector_hidden_size"], self.backbone.hidden_size, self.config["prefix_tokens"]).to(self.backbone.device)
        states = torch.load(root / "adapters.pt", map_location="cpu", weights_only=True)
        self.predictor.load_state_dict(states["predictor"])
        self.projector.load_state_dict(states["projector"])
        self.predictor.eval()
        self.projector.eval()
        self._lock = threading.Lock()

    @torch.inference_mode()
    def reply(self, messages):
        with self._lock:
            context = self.backbone.features([messages])
            source = self.backbone.features([[{"role": "user", "content": messages[-1]["content"]}]])
            target = self.predictor(torch.as_tensor(context, device=self.backbone.device), torch.as_tensor(source, device=self.backbone.device))["vector"][0]
            reply = self.backbone.generate(messages, target, self.projector,
                max_new_tokens=self.config["max_reply_tokens"], do_sample=self.config["do_sample"],
                temperature=self.config["temperature"], top_p=self.config["top_p"])
            generated_features = self.backbone.features([[{"role": "user", "content": reply}]])
            reply_vector = self.predictor.utterance(torch.as_tensor(generated_features, device=self.backbone.device))["vector"][0].cpu().numpy()
            nearest = self.index.nearest(reply_vector)
            return {"reply": reply, "meme_id": nearest["meme_id"]}
