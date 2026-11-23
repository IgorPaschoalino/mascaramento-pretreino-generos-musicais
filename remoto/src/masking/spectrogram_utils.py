"""
spectrogram_utils.py
====================
Remontagem correta de patches -> espectrograma no formato de entrada do AST.

Motivacao (sessao 26/08): o collator original remontava o espectrograma com
`input_values.view(B, -1, 128)`, que NAO e a inversa de
`preprocessing.spectrogram_to_patches()`. O resultado tinha 0% das posicoes
corretas — os valores certos em lugares errados, com os eixos de tempo e
frequencia interleavados. Este modulo implementa a inversa de verdade.

Convencoes:
  - `spectrogram_to_patches` recebe (n_mels, T) e devolve (n_patches, P, P)
    em ordem row-major sobre a grade (n_rows=blocos de frequencia,
    n_cols=blocos de tempo): indice do patch = r * n_cols + c.
  - O AST do HuggingFace espera `input_values` com shape (B, max_length,
    num_mel_bins) = (B, 1024, 128), isto e, (tempo, frequencia).
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def patches_to_spectrogram(patches: torch.Tensor,
                           grid_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Inversa exata de `spectrogram_to_patches`.

    Args:
        patches: (B, n_patches, P, P) — patches em ordem row-major sobre a
                 grade (n_rows, n_cols).
        grid_shape: (n_rows, n_cols) — blocos de frequencia x blocos de tempo.

    Returns:
        (B, n_mels, T) — espectrograma no layout (frequencia, tempo).
    """
    B, n_patches, ph, pw = patches.shape
    n_rows, n_cols = grid_shape
    if n_rows * n_cols != n_patches:
        raise ValueError(
            f"grid_shape {grid_shape} implica {n_rows * n_cols} patches, "
            f"mas foram recebidos {n_patches}."
        )

    # (B, n_rows, n_cols, ph, pw)
    x = patches.view(B, n_rows, n_cols, ph, pw)
    # -> (B, n_rows, ph, n_cols, pw)  (intercala o eixo de freq. dentro do bloco)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    # -> (B, n_rows*ph, n_cols*pw) = (B, n_mels, T)
    x = x.view(B, n_rows * ph, n_cols * pw)
    return x


def patches_to_ast_input(patches: torch.Tensor,
                         grid_shape: Tuple[int, int],
                         target_frames: int = 1024) -> torch.Tensor:
    """
    Remonta os patches e devolve no layout exato esperado pelo AST:
    (B, target_frames, n_mels), isto e, (tempo, frequencia), com padding
    (ou truncamento) no eixo temporal.

    Args:
        patches: (B, n_patches, P, P).
        grid_shape: (n_rows, n_cols).
        target_frames: comprimento temporal esperado pelo AST (1024).

    Returns:
        (B, target_frames, n_mels)
    """
    spec = patches_to_spectrogram(patches, grid_shape)   # (B, n_mels, T)
    spec = spec.transpose(1, 2).contiguous()             # (B, T, n_mels)

    T = spec.shape[1]
    if T < target_frames:
        spec = F.pad(spec, (0, 0, 0, target_frames - T))  # padding no fim do tempo
    elif T > target_frames:
        spec = spec[:, :target_frames, :]
    return spec


def ast_input_to_patches(spec: torch.Tensor,
                         grid_shape: Tuple[int, int],
                         patch_size: int = 16) -> torch.Tensor:
    """
    Caminho inverso: (B, T, n_mels) -> (B, n_patches, P, P), na mesma ordem
    row-major usada por `spectrogram_to_patches`. Util para montar o alvo de
    reconstrucao do MSM a partir do espectrograma limpo.
    """
    n_rows, n_cols = grid_shape
    x = spec[:, :n_cols * patch_size, :n_rows * patch_size]   # (B, T', n_mels')
    x = x.transpose(1, 2).contiguous()                        # (B, n_mels', T')
    B = x.shape[0]
    x = x.view(B, n_rows, patch_size, n_cols, patch_size)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    return x.view(B, n_rows * n_cols, patch_size, patch_size)
