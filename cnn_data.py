"""
cnn_data.py
===========
Pipeline de dados tensorizado para os experimentos com CNN.

Por que existe
--------------
A primeira versao usava `DataLoader` + collator sobre o dataset do HuggingFace,
como o pipeline do AST. Medido em execucao: GPU a 34% de utilizacao e o
processo consumindo 217% de CPU - ou seja, **o treino era limitado pela CPU**,
nao pela GPU. A causa e que o collator remonta o espectrograma a partir dos
patches em numpy a cada batch e a cada epoca, e essa remontagem custa mais que
o forward/backward de uma CNN de 1,2M parametros. A 26,4 min por execucao, as
25 execucoes planejadas levariam ~17 horas.

Duas mudancas eliminam o gargalo:

  1. **Materializar uma vez.** Os patches de cada split viram um unico tensor
     na RAM. Para o fine-tuning, o espectrograma limpo - que e sempre o mesmo -
     e remontado uma unica vez, e nao 40 vezes (uma por epoca).
  2. **Remontar na GPU.** No pre-treino MSM a remontagem depende da mascara,
     entao nao da para pre-computar; mas `patches_to_spectrogram` e so
     reshape/permute em torch, entao roda no device. So as mascaras (barato:
     496 scores por exemplo) continuam em numpy.

O resultado numerico e identico - a remontagem e deterministica e independente
de device.
"""

import numpy as np
import torch
from datasets import DatasetDict

from masking.spectrogram_utils import patches_to_spectrogram
from masking.patch_mask_collator import select_masked, cache_strategy_of


class TensorSplit:
    """Um split materializado: patches, scores, rotulos e grade."""

    def __init__(self, split):
        self.patches = torch.from_numpy(
            np.stack([np.asarray(e, dtype=np.float32) for e in split['input_values']]))
        self.scores = np.stack(
            [np.asarray(e, dtype=np.float64) for e in split['patch_scores']])
        self.labels = torch.tensor([int(v) for v in split['label']], dtype=torch.long)
        self.grid = tuple(int(x) for x in np.asarray(split[0]['grid_shape']))
        self._spec = None

    def __len__(self):
        return self.patches.shape[0]

    @property
    def spec(self):
        """Espectrograma limpo (N, 1, n_mels, T), remontado uma unica vez."""
        if self._spec is None:
            self._spec = patches_to_spectrogram(self.patches, self.grid).unsqueeze(1)
        return self._spec

    def drop_spec(self):
        """Libera o cache do espectrograma limpo (~355 MB no split de treino)."""
        self._spec = None


def load_splits(strategy, max_examples=None):
    ds = DatasetDict.load_from_disk(
        'cache/gtzan-%s-p16-s42' % cache_strategy_of(strategy)).with_format('numpy')
    if max_examples:
        ds = DatasetDict({k: v.select(range(min(max_examples, len(v))))
                          for k, v in ds.items()})
    return {k: TensorSplit(ds[k]) for k in ('train', 'validation', 'test')}


def iter_batches(n, batch_size, rng=None):
    """Indices de batch. `rng` embaralha; None mantem a ordem original."""
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


class MaskSampler:
    """Sorteia quais patches mascarar, pela estrategia da condicao.

    `uniform=True` e o baseline `random`: permutacao uniforme, ignorando os
    scores. Caso contrario o sorteio e ponderado pelos scores da estrategia
    (spectral_energy / onset / combined), que ja vieram pre-computados no cache.
    `strategy`, quando dada, prevalece (necessaria para inverse_energy e
    specaugment -- ver masking.patch_mask_collator.select_masked).
    """

    def __init__(self, mask_ratio=0.75, mask_choice='weighted_random',
                 temperature=0.1, seed=42, uniform=False, strategy=None, grid=None):
        self.mask_ratio = mask_ratio
        self.mask_choice = mask_choice
        self.temperature = temperature
        self.strategy = strategy or ('random' if uniform else 'scored')
        self.grid = grid
        self.rng = np.random.default_rng(seed)

    def masks_for(self, scores_batch):
        """(B, n_patches) bool: True onde mascarar."""
        m = np.zeros(scores_batch.shape, dtype=bool)
        n = scores_batch.shape[1]
        for i in range(scores_batch.shape[0]):
            m[i, select_masked(self.strategy, scores_batch[i], n, self.mask_ratio,
                               self.rng, self.grid, self.mask_choice,
                               self.temperature)] = True
        return torch.from_numpy(m)


def msm_batch(split, idx, sampler, device):
    """Monta um batch do estagio 1, remontando o espectrograma no device."""
    cp = split.patches[idx].to(device, non_blocking=True)
    mk = sampler.masks_for(split.scores[idx]).to(device, non_blocking=True)
    masked = cp.masked_fill(mk.unsqueeze(-1).unsqueeze(-1), 0.0)
    x = patches_to_spectrogram(masked, split.grid).unsqueeze(1)
    return x, cp, mk


def clean_batch(split, idx, device):
    """Batch do estagio 2 / avaliacao: fatia do espectrograma ja remontado."""
    return (split.spec[idx].to(device, non_blocking=True),
            split.labels[idx].to(device, non_blocking=True))
