"""
v2_common.py
============
Infraestrutura compartilhada da reformulacao metodologica v2 (sessao 26/08).

Duas correcoes em relacao ao pipeline original:

  1. Remontagem CORRETA dos patches no espectrograma (ver
     masking/spectrogram_utils.py). O pipeline antigo entregava ao AST os
     valores certos em posicoes erradas (0% de acerto posicional).

  2. Protocolo de DOIS ESTAGIOS, fiel a Belfathi et al. (2025):
       Estagio 1 (MSM): mascara patches pela estrategia e treina o encoder a
         RECONSTRUIR o conteudo original dos patches mascarados (MSE apenas
         nas posicoes mascaradas). E aqui que a seletividade tem mecanismo
         real para ajudar - o pretexto passa a focar conteudo saliente.
       Estagio 2: classificacao em espectrograma LIMPO, protocolo identico
         para todas as estrategias. A unica diferenca entre elas e o que foi
         mascarado no Estagio 1.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel
from sklearn.metrics import accuracy_score, f1_score

from masking.spectrogram_utils import patches_to_ast_input
from masking.patch_mask_collator import select_masked

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------

class CleanCollator:
    """Sem mascaramento nenhum: remonta o espectrograma completo + rotulo.

    Usado no Estagio 2 (fine-tuning) e na avaliacao - para as 4 estrategias,
    identicamente. Isso elimina o descasamento treino/teste que confundia a
    comparacao no protocolo antigo.
    """

    def __call__(self, examples):
        patches = torch.tensor(
            np.stack([np.asarray(e['input_values'], dtype=np.float32) for e in examples]))
        grid = tuple(int(x) for x in np.asarray(examples[0]['grid_shape']))
        spec = patches_to_ast_input(patches, grid, target_frames=1024)
        labels = torch.tensor([int(e['label']) for e in examples], dtype=torch.long)
        return {'input_values': spec, 'labels': labels}


class MSMCollator:
    """Mascaramento seletivo + alvo de reconstrucao (Estagio 1).

    Devolve:
      input_values   (B, 1024, 128) - espectrograma com patches mascarados zerados
      clean_patches  (B, N, 16, 16) - alvo (conteudo original dos patches)
      mask           (B, N) bool    - True onde foi mascarado (loss so aqui)
    """

    def __init__(self, mask_ratio=0.75, mask_choice='weighted_random',
                 temperature=0.1, seed=42, uniform=False, strategy=None):
        self.mask_ratio = mask_ratio
        self.mask_choice = mask_choice
        self.temperature = temperature
        # `strategy` decide o sorteio (ver select_masked); sem ela, vale o
        # comportamento antigo: uniform=True => random, senao ponderado.
        self.strategy = strategy or ('random' if uniform else 'scored')
        self.rng = np.random.default_rng(seed)

    def __call__(self, examples):
        grid = tuple(int(x) for x in np.asarray(examples[0]['grid_shape']))
        clean, masks = [], []
        for e in examples:
            p = np.asarray(e['input_values'], dtype=np.float32)
            s = np.asarray(e['patch_scores'], dtype=np.float64)
            m = np.zeros(len(p), dtype=bool)
            m[select_masked(self.strategy, s, len(p), self.mask_ratio, self.rng,
                            grid, self.mask_choice, self.temperature)] = True
            clean.append(p)
            masks.append(m)

        clean_t = torch.tensor(np.stack(clean))                    # (B, N, 16, 16)
        mask_t = torch.tensor(np.stack(masks))                     # (B, N)

        masked = clean_t.masked_fill(mask_t.unsqueeze(-1).unsqueeze(-1), 0.0)
        spec = patches_to_ast_input(masked, grid, target_frames=1024)
        return {'input_values': spec, 'clean_patches': clean_t, 'mask': mask_t}


def make_msm_collator(strategy, mask_ratio, mask_choice, temperature, seed):
    """`random` = uniforme; `specaugment` = faixas; demais = ponderado pelos
    scores (`inverse_energy` pelo inverso da energia)."""
    return MSMCollator(mask_ratio=mask_ratio, mask_choice=mask_choice,
                       temperature=temperature, seed=seed, strategy=strategy)


# ---------------------------------------------------------------------------
# Modelo do Estagio 1: AST + cabeca de reconstrucao
# ---------------------------------------------------------------------------

class ASTForMaskedSpectrogramModeling(nn.Module):
    """Encoder AST + decoder linear leve (estilo SSAST/MAE).

    A grade de tokens do AST e (12 freq x 101 tempo) por causa do stride 10
    sobreposto; a grade de scoring e (8 x 62), nao sobreposta. Reamostramos a
    grade de tokens para a grade de scoring (bilinear) e projetamos cada token
    nos 16x16=256 pixels do seu patch.
    """

    def __init__(self, grid_shape, checkpoint=AST_CHECKPOINT, patch_size=16):
        super().__init__()
        self.ast = ASTModel.from_pretrained(checkpoint)
        cfg = self.ast.config
        self.f_dim = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
        self.t_dim = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.patch_size = patch_size
        h = cfg.hidden_size
        self.decoder = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, patch_size * patch_size))

    def forward(self, input_values, clean_patches, mask):
        hs = self.ast(input_values).last_hidden_state          # (B, 2+F*T, C)
        tok = hs[:, 2:, :]                                     # (B, F*T, C)
        B, N, C = tok.shape
        g = tok.transpose(1, 2).reshape(B, C, self.f_dim, self.t_dim)
        g = F.interpolate(g, size=self.grid_shape, mode='bilinear', align_corners=False)
        g = g.flatten(2).transpose(1, 2)                        # (B, n_patches, C)

        P = self.patch_size
        pred = self.decoder(g).view(B, -1, P, P)                # (B, n_patches, P, P)

        per_patch = ((pred - clean_patches) ** 2).mean(dim=(2, 3))   # (B, n_patches)
        denom = mask.sum().clamp(min=1)
        loss = (per_patch * mask.float()).sum() / denom
        return loss


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }
