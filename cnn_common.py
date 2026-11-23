"""
cnn_common.py
=============
Backbone convolucional para a replicacao de arquitetura (sessao 28/08).

Por que uma CNN, e por que agora
--------------------------------
Todos os experimentos ate aqui usaram o AST (Audio Spectrogram Transformer)
como encoder, e todos deram resultado nulo: as quatro estrategias de
mascaramento colapsam na mesma acuracia, e no mesmo valor do controle sem MSM.
Existe uma explicacao alternativa ao "mascaramento seletivo nao funciona" que o
desenho atual nao consegue descartar:

  **O AST ja vem pre-treinado no AudioSet (~2 milhoes de clipes).** Vinte
  epocas de MSM sobre 699 clipes do GTZAN nao tem como mover materialmente um
  encoder desse porte. Se o estagio 1 nao muda o encoder, entao *qualquer*
  escolha de mascaramento produz o mesmo estagio 2 - e o resultado nulo e uma
  propriedade do ponto de partida, nao do metodo.

Uma CNN treinada do zero testa exatamente isso: sem pre-treino massivo, o MSM
e a unica fonte de representacao antes do fine-tuning, entao a escolha do que
mascarar passa a ter espaco real para importar. Se as estrategias continuarem
empatadas aqui, o resultado nulo e do metodo; se separarem, era saturacao do
AST.

Alinhamento exato patch <-> feature
-----------------------------------
Ha um segundo ganho, metodologico. No AST a grade de tokens e 12x101 (stride 10
sobreposto) enquanto a grade de scoring/mascaramento e 8x62 (nao sobreposta), e
`ASTForMaskedSpectrogramModeling` precisa interpolar bilinearmente de uma para
a outra - o alvo de reconstrucao chega borrado e deslocado. Esta CNN faz
downsample por um fator de exatamente 16 (quatro maxpools 2x2), entao o mapa de
features tem shape 8x62: **uma posicao de feature por patch, sem interpolacao
nenhuma**. O pretexto de reconstrucao fica limpo.
"""

import numpy as np
import torch
import torch.nn as nn

from masking.spectrogram_utils import patches_to_spectrogram
from masking.patch_mask_collator import weighted_random_order, top_n_order


# ---------------------------------------------------------------------------
# Collators (layout de CNN: (B, 1, n_mels, T), sem padding para 1024)
# ---------------------------------------------------------------------------

class CNNCleanCollator:
    """Espectrograma completo + rotulo. Usado no estagio 2 e na avaliacao."""

    def __call__(self, examples):
        patches = torch.tensor(
            np.stack([np.asarray(e['input_values'], dtype=np.float32) for e in examples]))
        grid = tuple(int(x) for x in np.asarray(examples[0]['grid_shape']))
        spec = patches_to_spectrogram(patches, grid).unsqueeze(1)   # (B,1,mel,T)
        labels = torch.tensor([int(e['label']) for e in examples], dtype=torch.long)
        return {'x': spec, 'labels': labels}


class CNNMSMCollator:
    """Mascaramento seletivo + alvo de reconstrucao (estagio 1).

    Devolve:
      x              (B, 1, n_mels, T)  espectrograma com os patches zerados
      clean_patches  (B, N, 16, 16)     alvo (conteudo original)
      mask           (B, N) bool        True onde foi mascarado (loss so aqui)
    """

    def __init__(self, mask_ratio=0.75, mask_choice='weighted_random',
                 temperature=0.1, seed=42, uniform=False):
        self.mask_ratio = mask_ratio
        self.mask_choice = mask_choice
        self.temperature = temperature
        self.uniform = uniform          # True => baseline `random`
        self.rng = np.random.default_rng(seed)

    def _select(self, scores):
        n = len(scores)
        k = max(1, int(round(n * self.mask_ratio)))
        if self.uniform:
            order = self.rng.permutation(n)
        elif self.mask_choice == 'weighted_random':
            order = weighted_random_order(scores, self.rng, self.temperature)
        else:
            order = top_n_order(scores)
        return order[:k]

    def __call__(self, examples):
        clean, masks = [], []
        for e in examples:
            p = np.asarray(e['input_values'], dtype=np.float32)
            s = np.asarray(e['patch_scores'], dtype=np.float64)
            m = np.zeros(len(p), dtype=bool)
            m[self._select(s)] = True
            clean.append(p)
            masks.append(m)

        clean_t = torch.tensor(np.stack(clean))                  # (B, N, 16, 16)
        mask_t = torch.tensor(np.stack(masks))                   # (B, N)
        grid = tuple(int(x) for x in np.asarray(examples[0]['grid_shape']))
        masked = clean_t.masked_fill(mask_t.unsqueeze(-1).unsqueeze(-1), 0.0)
        spec = patches_to_spectrogram(masked, grid).unsqueeze(1)
        return {'x': spec, 'clean_patches': clean_t, 'mask': mask_t}


def make_cnn_msm_collator(strategy, mask_ratio, mask_choice, temperature, seed):
    return CNNMSMCollator(mask_ratio=mask_ratio, mask_choice=mask_choice,
                          temperature=temperature, seed=seed,
                          uniform=(strategy == 'random'))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SpecCNNEncoder(nn.Module):
    """VGG-like de 4 blocos. Downsample total = 16, igual ao patch_size.

    Entrada (B,1,128,992) -> saida (B,256,8,62), que e exatamente a grade de
    patches (8x62). Cada vetor de 256 canais corresponde a um patch 16x16.
    ~1,2M parametros, contra ~86M do AST.
    """

    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        chans = (1,) + tuple(widths)
        self.blocks = nn.Sequential(
            *[_block(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.out_channels = widths[-1]

    def forward(self, x):
        return self.blocks(x)


class CNNForMaskedSpectrogramModeling(nn.Module):
    """Estagio 1: encoder + decoder 1x1 que reconstroi cada patch mascarado.

    O decoder e deliberadamente raso (uma conv 1x1) para que a pressao do
    pretexto recaia sobre o encoder, e nao sobre a cabeca - a mesma logica do
    MAE/SSAST, e o que torna a comparacao com o AST justa.
    """

    def __init__(self, patch_size=16, widths=(32, 64, 128, 256)):
        super().__init__()
        self.encoder = SpecCNNEncoder(widths)
        self.patch_size = patch_size
        self.decoder = nn.Conv2d(self.encoder.out_channels,
                                 patch_size * patch_size, 1)

    def forward(self, x, clean_patches, mask):
        f = self.encoder(x)                             # (B, C, 8, 62)
        pred = self.decoder(f)                          # (B, 256, 8, 62)
        B, _, H, W = pred.shape
        P = self.patch_size
        # (B,P*P,H,W) -> (B,H,W,P,P) -> (B,H*W,P,P), row-major igual aos patches
        pred = pred.permute(0, 2, 3, 1).reshape(B, H * W, P, P)
        per_patch = ((pred - clean_patches) ** 2).mean(dim=(2, 3))
        denom = mask.sum().clamp(min=1)
        return (per_patch * mask.float()).sum() / denom


class CNNClassifier(nn.Module):
    """Estagio 2: mesmo encoder + pooling global + linear."""

    def __init__(self, num_labels, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        self.encoder = SpecCNNEncoder(widths)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.encoder.out_channels, num_labels)

    def forward(self, x):
        f = self.pool(self.encoder(x)).flatten(1)
        return self.fc(self.drop(f))
