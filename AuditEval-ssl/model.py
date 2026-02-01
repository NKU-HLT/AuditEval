import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScorePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


class MosPredictor(nn.Module):
    def __init__(self, upstream_model, upstream_output_dim,
                 quality_mlp=None, relevance_mlp=None, faithfulness_mlp=None):
        super().__init__()
        self.upstream_model = upstream_model
        self.upstream_feat_dim = upstream_output_dim

        self.norm_ori_audio = nn.LayerNorm(self.upstream_feat_dim)
        self.norm_tar_audio = nn.LayerNorm(self.upstream_feat_dim)
        self.norm_ori_text = nn.LayerNorm(self.upstream_feat_dim)
        self.norm_tar_text = nn.LayerNorm(self.upstream_feat_dim)

        combined_input_dim = self.upstream_feat_dim * 4

        self.quality_mlp = quality_mlp if quality_mlp is not None else ScorePredictor(combined_input_dim)
        self.relevance_mlp = relevance_mlp if relevance_mlp is not None else ScorePredictor(combined_input_dim)
        self.faithfulness_mlp = faithfulness_mlp if faithfulness_mlp is not None else ScorePredictor(combined_input_dim)

    def forward(self, ori_wavs, ori_prompts, tar_wavs, tar_prompts):
        device = ori_wavs.device

        ori_audio_embed = self.upstream_model.get_audio_embedding_from_data(ori_wavs, use_tensor=True).to(device)
        tar_audio_embed = self.upstream_model.get_audio_embedding_from_data(tar_wavs, use_tensor=True).to(device)
        ori_text_embed = self.upstream_model.get_text_embedding(ori_prompts, use_tensor=True).to(device)
        tar_text_embed = self.upstream_model.get_text_embedding(tar_prompts, use_tensor=True).to(device)

        ori_audio_embed = self.norm_ori_audio(ori_audio_embed)
        tar_audio_embed = self.norm_tar_audio(tar_audio_embed)
        ori_text_embed = self.norm_ori_text(ori_text_embed)
        tar_text_embed = self.norm_tar_text(tar_text_embed)

        combined_feat = torch.cat([
            ori_audio_embed,
            ori_text_embed,
            tar_audio_embed,
            tar_text_embed
        ], dim=1)

        quality_score = self.quality_mlp(combined_feat) if self.quality_mlp is not None else None
        relevance_score = self.relevance_mlp(combined_feat) if self.relevance_mlp is not None else None
        faithfulness_score = self.faithfulness_mlp(combined_feat) if self.faithfulness_mlp is not None else None

        return quality_score, relevance_score, faithfulness_score


