"""
remoto_common.py
================
Nucleo compartilhado da execucao remota (FMA-small + Homburg) do experimento
de mascaramento seletivo.

Herda o protocolo de DOIS ESTAGIOS validado no GTZAN (v2_common.py da sessao
26/08), com uma unica mudanca de ENGENHARIA (nao de metodo):

  O cache do GTZAN gravava uma copia inteira do dataset por estrategia
  (4 copias, em float64), porque cada copia carregava sua propria coluna
  `patch_scores` -- e os patches, que sao a parte enorme, eram identicos entre
  elas. Aqui gravamos UM cache por dataset, em float32, com as TRES colunas de
  score num mesmo registro (`score_energy`, `score_onset`, `score_combined`);
  `random` nao precisa de coluna nenhuma (mascaramento uniforme).

  Cada estrategia le no Estagio 1 exatamente a mesma sequencia de numeros que o
  pipeline do GTZAN lhe entregava (ver `scores_for`, e o teste de equivalencia
  em testar_pacote.py). Alphas fora do valor canonico 0.5 continuam disponiveis,
  derivados das duas componentes por `combinar_scores`.

  Ganho: ~40 GB -> ~5 GB para os dois datasets. E um ganho de RIGOR, de quebra:
  os patches passam a ser identicos entre estrategias por CONSTRUCAO. Antes
  eram identicos apenas por coincidencia -- quatro processos separados sorteando
  o mesmo crop aleatorio por partirem da mesma semente.

O que NAO muda em relacao ao GTZAN, de proposito, para manter a comparacao
entre datasets honesta: batch_size 8, mask_ratio 0.75, temperature 0.1,
alpha 0.5, 30 epocas de MSM, 15 de fine-tuning, lr 5e-5, fp16.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel, ASTForAudioClassification
from sklearn.metrics import accuracy_score, f1_score

from masking.spectrogram_utils import patches_to_ast_input
from masking.patch_mask_collator import select_masked

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
EPSILON = 1e-6
# As 4 estrategias do estudo original + 2 de referencia (ciclo de 15/10):
#   inverse_energy -- espelho de spectral_energy (mascara os patches de MENOR
#                     energia); teste de sanidade do criterio de selecao.
#   specaugment    -- faixas inteiras de tempo/frequencia, o padrao da area.
ESTRATEGIAS_ORIGINAIS = ['random', 'spectral_energy', 'onset', 'combined']
ESTRATEGIAS_NOVAS = ['inverse_energy', 'specaugment']
STRATEGIES = ESTRATEGIAS_ORIGINAIS + ESTRATEGIAS_NOVAS


# ---------------------------------------------------------------------------
# Registro de datasets
# ---------------------------------------------------------------------------

FMA_GENRES = [
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock",
]

HOMBURG_GENRES = [
    "alternative", "blues", "electronic", "folkcountry", "funksoulrnb",
    "jazz", "pop", "raphiphop", "rock",
]

DATASETS = {
    'fma': {
        'genres': FMA_GENRES,
        'cache': 'cache_remoto/fma-p16',
        # Splits NATIVOS do repositorio HuggingFace (6400/800/800). Mantidos
        # como publicados: sao reproduziveis por terceiros sem reimplementar
        # nossa divisao, e os 800 exemplos de teste dao o melhor poder
        # estatistico do projeto (MDE ~6.9 pp vs 10.4 pp do GTZAN).
        'split': 'nativo',
    },
    'homburg': {
        'genres': HOMBURG_GENRES,
        'cache': 'cache_remoto/homburg-p16',
        # O repositorio publica tudo num unico split, entao dividimos nos.
        # 70/15/15 = MESMA proporcao usada no GTZAN, para que a comparacao
        # entre datasets nao carregue diferenca de protocolo de divisao.
        # (80/10/10 daria test=188 e MDE de 14 pp, PIOR que o GTZAN.)
        'split': (0.70, 0.15),
    },
}


def genres_of(dataset):
    if dataset not in DATASETS:
        raise ValueError('dataset desconhecido: %s (opcoes: %s)'
                         % (dataset, list(DATASETS)))
    return DATASETS[dataset]['genres']


def cache_path_of(dataset):
    """Caminho do cache do dataset. Se `localizar_dados.py` achou um cache
    pronto em outro lugar da maquina (de uma corrida anterior), ele grava o
    caminho em cache_remoto/onde_esta.json e o cache e usado de la, sem copiar
    nem reprocessar."""
    import json, os
    mapa = os.path.join('cache_remoto', 'onde_esta.json')
    if os.path.exists(mapa):
        try:
            with open(mapa, encoding='utf-8') as f:
                alt = json.load(f).get(dataset)
            if alt and os.path.isdir(os.path.join(alt, 'test')):
                return alt
        except (OSError, ValueError):
            pass
    return DATASETS[dataset]['cache']


# ---------------------------------------------------------------------------
# Derivacao de scores a partir das colunas base
# ---------------------------------------------------------------------------

def _minmax(arr):
    """Identico ao `minmax` interno de CombinedScoring.score_patches."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


# Alpha para o qual a coluna `score_combined` e pre-computada no cache. E o
# valor canonico do experimento; outros alphas sao derivados (ver `scores_for`).
STORED_ALPHA = 0.5


def combinar_scores(score_energy, score_onset, alpha=STORED_ALPHA):
    """Reproduz `CombinedScoring(alpha).score_patches(patches, normalize=False)`
    a partir das duas componentes.

    ATENCAO aos dtypes, que NAO sao um detalhe: os scorers originais devolvem
    `energy` em float32 (sqrt/mean sobre patches float32) e `onset` em float64
    (a list-comprehension com `float()` promove). O `minmax` arredonda de forma
    diferente nas duas precisoes, e uma divergencia de ~1e-8 nos scores se
    propaga por `scores**(1/temperature)` = `scores**10` ate mudar a mascara
    sorteada. Reimpomos os dtypes originais para que a combinacao seja
    bit-a-bit igual a da classe original.
    """
    e32 = np.asarray(score_energy, dtype=np.float32)
    o64 = np.asarray(score_onset, dtype=np.float64)
    return alpha * _minmax(e32) + (1.0 - alpha) * _minmax(o64) + EPSILON


def scores_for(strategy, ex, alpha=STORED_ALPHA):
    """Scores que o Estagio 1 usa para escolher a mascara, no formato em que o
    pipeline do GTZAN os entregava ao collator.

    Por que float64 na saida: o MSMCollator do GTZAN fazia
    `np.asarray(e['patch_scores'], dtype=np.float64)`, e o que ele recebia do
    cache ja vinha arredondado para float32 -- porque `datasets` rebaixa
    float64 para float32 em `with_format('numpy')`, independentemente do dtype
    declarado no schema (verificado na versao 4.8.5). Ou seja, o que o
    experimento do GTZAN realmente consumiu foi `float64(float32(score))`, e e
    isso que reproduzimos aqui.

    Para `combined` no alpha canonico usamos a coluna PRE-COMPUTADA do cache, em
    vez de recombinar: o pipeline antigo arredondava para float32 DEPOIS de
    combinar, e recombinar a partir de componentes ja arredondadas daria um
    resultado ligeiramente diferente (~1e-8). Guardar a coluna pronta custa
    ~2 KB por exemplo e elimina a questao.

    Devolve None para `random` e `specaugment`, que nao usam scores.
    """
    if strategy in ('random', 'specaugment'):
        return None
    if strategy in ('spectral_energy', 'inverse_energy'):
        # inverse_energy recebe a ENERGIA; a inversao e feita em select_masked.
        return np.asarray(ex['score_energy'], dtype=np.float64)
    if strategy == 'onset':
        return np.asarray(ex['score_onset'], dtype=np.float64)
    if strategy == 'combined':
        if abs(alpha - STORED_ALPHA) < 1e-12 and 'score_combined' in ex:
            return np.asarray(ex['score_combined'], dtype=np.float64)
        # Alpha nao-canonico: derivado das componentes. Divergencia em relacao a
        # coluna pre-computada e da ordem de 1e-8 (abaixo do epsilon do float32).
        return np.asarray(combinar_scores(ex['score_energy'], ex['score_onset'], alpha),
                          dtype=np.float64)
    raise ValueError('estrategia desconhecida: %s' % strategy)


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------

class CleanCollator:
    """Sem mascaramento: remonta o espectrograma completo + rotulo.

    Usado no Estagio 2 (fine-tuning) e na avaliacao, identicamente para as 4
    estrategias -- e o que elimina o descasamento treino/teste que confundia a
    comparacao no protocolo antigo.
    """

    def __call__(self, examples):
        patches = torch.from_numpy(
            np.stack([np.asarray(e['input_values'], dtype=np.float32) for e in examples]))
        grid = tuple(int(x) for x in np.asarray(examples[0]['grid_shape']))
        spec = patches_to_ast_input(patches, grid, target_frames=1024)
        labels = torch.tensor([int(e['label']) for e in examples], dtype=torch.long)
        return {'input_values': spec, 'labels': labels}


class MSMCollator:
    """Mascaramento seletivo + alvo de reconstrucao (Estagio 1).

    Diferenca em relacao ao MSMCollator do GTZAN: os scores sao DERIVADOS das
    colunas base em vez de lidos de uma coluna `patch_scores` pre-computada.
    O resultado numerico e o mesmo (ver `scores_for`).

    Devolve:
      input_values   (B, 1024, 128) - espectrograma com patches mascarados zerados
      clean_patches  (B, N, 16, 16) - alvo (conteudo original dos patches)
      mask           (B, N) bool    - True onde foi mascarado (loss so aqui)
    """

    def __init__(self, strategy, mask_ratio=0.75, mask_choice='weighted_random',
                 temperature=0.1, alpha=0.5, seed=42):
        self.strategy = strategy
        self.mask_ratio = mask_ratio
        self.mask_choice = mask_choice
        self.temperature = temperature
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def __call__(self, examples):
        grid = tuple(int(x) for x in np.asarray(examples[0]['grid_shape']))
        clean, masks = [], []
        for e in examples:
            p = np.asarray(e['input_values'], dtype=np.float32)
            s = scores_for(self.strategy, e, self.alpha)
            m = np.zeros(len(p), dtype=bool)
            m[select_masked(self.strategy, s, len(p), self.mask_ratio, self.rng,
                            grid, self.mask_choice, self.temperature)] = True
            clean.append(p)
            masks.append(m)

        clean_t = torch.from_numpy(np.stack(clean))               # (B, N, 16, 16)
        mask_t = torch.from_numpy(np.stack(masks))                # (B, N)

        masked = clean_t.masked_fill(mask_t.unsqueeze(-1).unsqueeze(-1), 0.0)
        spec = patches_to_ast_input(masked, grid, target_frames=1024)
        return {'input_values': spec, 'clean_patches': clean_t, 'mask': mask_t}


# ---------------------------------------------------------------------------
# Modelo do Estagio 1: AST + cabeca de reconstrucao
# ---------------------------------------------------------------------------

class ASTForMaskedSpectrogramModeling(nn.Module):
    """Encoder AST + decoder linear leve (estilo SSAST/MAE).

    Copia fiel da versao validada no GTZAN. A grade de tokens do AST e
    (12 freq x 101 tempo) por causa do stride 10 sobreposto; a grade de scoring
    e (8 x 62), nao sobreposta. Reamostramos a grade de tokens para a grade de
    scoring (bilinear) e projetamos cada token nos 16x16=256 pixels do patch.
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


def fresh_classifier(dataset):
    """Classificador AST novo, com a cabeca dimensionada para o dataset."""
    genres = genres_of(dataset)
    return ASTForAudioClassification.from_pretrained(
        AST_CHECKPOINT,
        num_labels=len(genres),
        label2id={g: i for i, g in enumerate(genres)},
        id2label={i: g for i, g in enumerate(genres)},
        ignore_mismatched_sizes=True,
    )


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
