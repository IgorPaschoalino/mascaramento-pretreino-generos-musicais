"""
patch_mask_collator.py
======================
Data collator para mascaramento seletivo de patches de espectrograma
no fine-tuning do SSAST.

Adaptado de:
    Belfathi et al. (2025) "Is Selective Masking A Key to Improving Domain
    Adaptation for Masked Language Model?" ICAIL 2025.
    https://github.com/ygorg/legal-masking  (custom_data_collator.py)

No trabalho original, o DataCollatorForTermSpecificMasking substitui o
mascaramento aleatório de tokens do BERT por um mascaramento ponderado
pelos scores de relevância de cada palavra. Aqui, aplicamos a mesma
lógica ao domínio de patches de espectrograma: em vez de mascarar patches
aleatoriamente (como no SSAST original), mascaramos proporcionalmente aos
scores de relevância acústica calculados pelo patch_scoring.py.

A interface é compatível com o Trainer da HuggingFace.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np
import torch

from masking.spectrogram_utils import patches_to_ast_input


# ---------------------------------------------------------------------------
# Função auxiliar: weighted_random (diretamente do código deles)
# ---------------------------------------------------------------------------

def weighted_random_order(scores: np.ndarray, rng: np.random.Generator,
                          temperature: float = 0.1) -> np.ndarray:
    """
    Ordena os índices dos patches de forma pseudo-aleatória ponderada
    pelos scores de relevância.

    Adaptado diretamente de DataCollatorForTermSpecificMasking.weighted_random_method
    em Belfathi et al. (2025). A única mudança é que aqui os elementos são
    patches (inteiros indexando a grade 2D) em vez de palavras.

    A temperatura controla o grau de seletividade:
    - temperature → 0: sempre mascara os patches de maior score (top-N)
    - temperature → ∞: mascaramento uniforme (equivalente ao aleatório)

    Args:
        scores: array 1D de scores não-negativos, um por patch.
        rng: gerador numpy para reprodutibilidade.
        temperature: controla a concentração da distribuição.

    Returns:
        array 1D de índices, ordenados do mais provável ao menos provável
        de ser mascarado.
    """
    # Softmax com temperatura: eleva cada score a (1/temperature)
    temp_scores = scores ** (1.0 / temperature)
    total = temp_scores.sum()
    if total == 0:
        norm_scores = np.ones(len(scores)) / len(scores)
    else:
        norm_scores = temp_scores / total

    # Amostragem sem reposição ponderada pelos scores normalizados
    idx_order = rng.choice(
        len(scores),
        size=len(scores),
        replace=False,
        p=norm_scores
    )
    return idx_order


def top_n_order(scores: np.ndarray) -> np.ndarray:
    """
    Ordena os índices do maior para o menor score (mascaramento determinístico).

    Adaptado de DataCollatorForTermSpecificMasking.top_n_method
    em Belfathi et al. (2025).

    Args:
        scores: array 1D de scores.

    Returns:
        array 1D de índices ordenados do maior ao menor score.
    """
    return np.argsort(scores)[::-1].copy()


# ---------------------------------------------------------------------------
# Estrategias de referencia (ciclo de 15/10): inverse_energy e specaugment
#
# Entram no estudo comparativo como pontos de referencia, nao como propostas:
#   - inverse_energy: espelho exato de spectral_energy (p ~ e^-10 em vez de
#     e^10). Teste de sanidade -- se ate mascarar o OPOSTO do criterio empata,
#     o criterio de selecao nao importa.
#   - specaugment: mascaramento estruturado em faixas inteiras de tempo e de
#     frequencia (Park et al., 2019), o padrao da area em audio.
# As 4 estrategias originais seguem consumindo a RNG exatamente como antes.
# ---------------------------------------------------------------------------

# Estrategias novas -> cache de onde leem os scores. inverse_energy usa os
# scores de energia; specaugment nao usa score nenhum (como random).
CACHE_STRATEGY = {'inverse_energy': 'spectral_energy', 'specaugment': 'random'}

# Larguras maximas das faixas, em patches. Grade (8 freq x 62 tempo), patch de
# 16 frames x 10 ms: 6 patches ~ 1 s de tempo, 2 patches = 32 bins Mel --
# proximos de T=100 frames e F=27 bins da politica LibriSpeech do SpecAugment.
SPECAUG_MAX_T = 6
SPECAUG_MAX_F = 2


def cache_strategy_of(strategy: str) -> str:
    """Nome da estrategia cujo cache (scores pre-computados) esta usa."""
    return CACHE_STRATEGY.get(strategy, strategy)


def inverse_energy_scores(energy_scores: np.ndarray) -> np.ndarray:
    """1/e: com temperature=0.1, p ~ e^-10, o espelho de spectral_energy.
    O piso 1e-6 limita a 1e60 apos a potencia -- sem overflow em float64."""
    return 1.0 / (np.asarray(energy_scores, dtype=np.float64) + 1e-6)


def specaugment_indices(grid_shape, k: int, rng: np.random.Generator,
                        max_t: int = SPECAUG_MAX_T,
                        max_f: int = SPECAUG_MAX_F) -> np.ndarray:
    """Indices de exatamente `k` patches, cobertos por faixas de tempo (todas
    as linhas de frequencia em `w` colunas) e de frequencia (todas as colunas
    em `h` linhas), sorteadas meio a meio ate cobrir `k`. A ultima faixa e
    truncada ao acaso, para que a proporcao mascarada seja a mesma das outras
    estrategias. Indice = linha * n_cols + coluna (row-major, ver
    spectrogram_utils)."""
    n_rows, n_cols = int(grid_shape[0]), int(grid_shape[1])
    seen = np.zeros(n_rows * n_cols, dtype=bool)
    out = []
    while len(out) < k:
        if rng.random() < 0.5:
            w = int(rng.integers(1, max_t + 1))
            c0 = int(rng.integers(0, n_cols - w + 1))
            idx = (np.arange(n_rows)[:, None] * n_cols
                   + np.arange(c0, c0 + w)[None, :]).ravel()
        else:
            h = int(rng.integers(1, max_f + 1))
            r0 = int(rng.integers(0, n_rows - h + 1))
            idx = (np.arange(r0, r0 + h)[:, None] * n_cols
                   + np.arange(n_cols)[None, :]).ravel()
        new = idx[~seen[idx]]
        rng.shuffle(new)
        seen[new] = True
        out.extend(new.tolist())
    return np.asarray(out[:k], dtype=np.int64)


def select_masked(strategy: str, scores, n: int, mask_ratio: float,
                  rng: np.random.Generator, grid_shape=None,
                  mask_choice: str = 'weighted_random',
                  temperature: float = 0.1) -> np.ndarray:
    """Indices a mascarar para qualquer estrategia. `scores` e ignorado por
    random/specaugment; para inverse_energy sao os scores de ENERGIA."""
    k = max(1, int(round(n * mask_ratio)))
    if strategy == 'random':
        return rng.permutation(n)[:k]
    if strategy == 'specaugment':
        if grid_shape is None:
            raise ValueError('specaugment precisa de grid_shape')
        return specaugment_indices(grid_shape, k, rng)
    s = np.asarray(scores, dtype=np.float64)
    if strategy == 'inverse_energy':
        s = inverse_energy_scores(s)
    if mask_choice == 'weighted_random':
        return weighted_random_order(s, rng, temperature)[:k]
    return top_n_order(s)[:k]


# ---------------------------------------------------------------------------
# Data Collator principal
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForSelectivePatchMasking:
    """
    Data collator que aplica mascaramento seletivo de patches de espectrograma
    para o fine-tuning do SSAST com a estratégia MSDM.

    Recebe um batch de dicionários com as chaves:
        - 'input_values': tensor (n_patches, patch_h, patch_w) — patches do espectrograma
        - 'patch_scores': lista/array 1D (n_patches,) — scores de relevância pré-calculados
          (produzidos pelo patch_scoring.py)

    Retorna um dicionário com:
        - 'input_values': patches originais
        - 'bool_masked_pos': máscara booleana (n_patches,), True = patch mascarado
        - 'labels': cópia dos patches originais (target de reconstrução)

    Compatível com o Trainer da HuggingFace.
    """

    mask_ratio: float = 0.75
    """Fração de patches a mascarar. Padrão SSAST = 0.75."""

    mask_choice_strategy: str = 'weighted_random'
    """Estratégia de seleção: 'weighted_random' ou 'top_n'."""

    temperature: float = 0.1
    """Temperatura para weighted_random. Menor = mais seletivo."""

    seed: int = 42
    """Semente para reprodutibilidade."""

    return_masked_pos: bool = False
    """Se True, mantém a chave 'bool_masked_pos' no retorno (usado para diagnóstico)."""

    correct_reassembly: bool = False
    """Se True, remonta os patches no espectrograma com a inversa CORRETA de
    `spectrogram_to_patches` (ver masking/spectrogram_utils.py). O padrão False
    preserva o comportamento (bugado) das sessões anteriores, para que os
    resultados antigos continuem reproduzíveis."""

    return_clean_patches: bool = False
    """Se True, devolve também 'clean_patches' (patches originais, sem zerar) e
    'grid_shape' — necessário para o alvo de reconstrução do MSM."""

    def __post_init__(self):
        if self.mask_choice_strategy not in ('weighted_random', 'top_n'):
            raise ValueError(
                f"mask_choice_strategy deve ser 'weighted_random' ou 'top_n', "
                f"recebido: {self.mask_choice_strategy}"
            )
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio deve estar em (0, 1), recebido: {self.mask_ratio}")
        self.rng = np.random.default_rng(seed=self.seed)

    def _select_patches_to_mask(self, scores: np.ndarray) -> np.ndarray:
        """
        Seleciona os índices dos patches a mascarar com base nos scores.

        Replica a lógica do Algorithm 1 de Belfathi et al. (2025):
        ordena os patches por relevância (weighted_random ou top_n) e
        seleciona os primeiros até atingir a proporção mask_ratio.

        Args:
            scores: array 1D (n_patches,) com scores de relevância.

        Returns:
            array 1D de índices dos patches selecionados para mascaramento.
        """
        n_patches = len(scores)
        n_to_mask = max(1, int(round(n_patches * self.mask_ratio)))

        # Ordenar patches por relevância (análogo ao `Sample(S)` do Algoritmo 1)
        if self.mask_choice_strategy == 'weighted_random':
            ordered_idx = weighted_random_order(scores, self.rng, self.temperature)
        else:
            ordered_idx = top_n_order(scores)

        # Selecionar os primeiros n_to_mask índices da ordenação
        selected = ordered_idx[:n_to_mask]
        return selected

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processa um batch de exemplos e gera as máscaras seletivas.

        Args:
            examples: lista de dicionários, cada um com:
                - 'input_values': array/tensor (n_patches, patch_h, patch_w)
                - 'patch_scores': array 1D (n_patches,) com scores de relevância

        Returns:
            dicionário com 'input_values', 'bool_masked_pos', 'labels'.
        """
        batch_patches = []
        batch_masks = []
        batch_grids = []

        for example in examples:
            patches = np.array(example['input_values'])   # (n_patches, H, W)
            # float64 obrigatorio: com temperature=0.1 calcula-se scores**10, e um
            # score no piso epsilon=1e-6 (patches silenciosos, so no `onset`) vira
            # 1e-60 -- representavel em float64, mas underflow para 0 em float32,
            # o que quebra rng.choice(replace=False) com 'Fewer non-zero entries in p'.
            scores = np.array(example['patch_scores'], dtype=np.float64)  # (n_patches,)

            n_patches = len(patches)

            if len(scores) != n_patches:
                raise ValueError(
                    f"Número de scores ({len(scores)}) difere do número de "
                    f"patches ({n_patches})."
                )

            # Selecionar patches a mascarar
            mask_idx = self._select_patches_to_mask(scores)

            # Construir máscara booleana (True = mascarado)
            bool_mask = np.zeros(n_patches, dtype=bool)
            bool_mask[mask_idx] = True

            batch_patches.append(patches)
            batch_masks.append(bool_mask)
            batch_grids.append(tuple(example['grid_shape']) if 'grid_shape' in example else None)

        
        # Converter para tensores PyTorch
        input_values = torch.tensor(np.stack(batch_patches), dtype=torch.float32) # (B, n_patches, H, W)
        bool_masked_pos = torch.tensor(np.stack(batch_masks), dtype=torch.bool)   # (B, n_patches)

        # -------------------------------------------------------------------
        # SIMULAÇÃO DO MASCARAMENTO SELETIVO (A Lógica Central do seu TCC)
        # Substitui os patches que devem ser mascarados por zero (silêncio)
        # -------------------------------------------------------------------
        B, n_patches, patch_h, patch_w = input_values.shape
        
        # Cria uma máscara expandida para aplicar nos tensores (B, n_patches, 1, 1)
        mask_expanded = bool_masked_pos.unsqueeze(-1).unsqueeze(-1)
        
        # Zera os patches selecionados pela estratégia seletiva (Combined/MSDM)
        input_values = input_values.masked_fill(mask_expanded, 0.0)

        # -------------------------------------------------------------------
        # RECONSTRUÇÃO DO ESPECTROGRAMA (Ajuste Dinâmico de Shape para o AST)
        # -------------------------------------------------------------------
        # Usamos o -1 para que o PyTorch calcule dinamicamente a dimensão do tempo
        # baseando-se estritamente no tamanho real do lote atual (seja 2 ou 8).
        if self.correct_reassembly:
            # Remontagem CORRETA (inversa de spectrogram_to_patches): devolve
            # (B, 1024, 128) = (tempo, frequência), preservando a estrutura
            # tempo-frequência real que o AST pré-treinado espera.
            grid = batch_grids[0]
            if grid is None:
                raise ValueError(
                    "correct_reassembly=True exige a coluna 'grid_shape' no dataset."
                )
            spectrogram_view = patches_to_ast_input(input_values, grid, target_frames=1024)
        else:
            # Comportamento legado (bugado) — mantido só para reprodutibilidade
            # dos experimentos anteriores a 26/08.
            spectrogram_view = input_values.view(B, -1, 128)

            if spectrogram_view.shape[1] != 1024:
                if spectrogram_view.shape[1] < 1024:
                    padding = torch.zeros((B, 1024 - spectrogram_view.shape[1], 128), dtype=torch.float32)
                    spectrogram_view = torch.cat([spectrogram_view, padding], dim=1)
                else:
                    spectrogram_view = spectrogram_view[:, :1024, :]

        # Criamos o dicionário do lote final adequado ao ASTForAudioClassification
        batch = {
            'input_values': spectrogram_view,       # [B, 1024, 128]
            'labels': torch.tensor([e['label'] for e in examples], dtype=torch.long)
        }

        # Mantém a chave da máscara ativa se a flag for True (necessário para o Diagnóstico)
        if self.return_masked_pos:
            batch['bool_masked_pos'] = bool_masked_pos

        if self.return_clean_patches:
            batch['clean_patches'] = torch.tensor(np.stack(batch_patches), dtype=torch.float32)
            batch['grid_shape'] = batch_grids[0]

        return batch


# ---------------------------------------------------------------------------
# Utilitário de diagnóstico (análogo ao demonstrate_data_collator deles)
# ---------------------------------------------------------------------------

def demonstrate_collator(collator: DataCollatorForSelectivePatchMasking,
                         dataset,
                         num_examples: int = 2) -> None:
    """
    Imprime exemplos do mascaramento aplicado pelo collator para inspeção
    visual rápida. Mostra quantos patches foram mascarados e a distribuição
    de scores entre mascarados e não-mascarados.

    Args:
        collator: instância de DataCollatorForSelectivePatchMasking.
        dataset: dataset HuggingFace com campos 'input_values' e 'patch_scores'.
        num_examples: quantos exemplos mostrar.
    """
    print("=" * 60)
    print("Diagnóstico do mascaramento seletivo de patches")
    print("=" * 60)

    collator.return_masked_pos = True

    samples = [dataset[i] for i in range(min(num_examples, len(dataset)))]
    batch = collator(samples)

    # VOLTA A DESATIVAR PARA NÃO QUEBRAR O TREINAMENTO DEPOIS
    collator.return_masked_pos = False

    for i, (mask, scores_raw) in enumerate(
        zip(batch['bool_masked_pos'], [np.array(s['patch_scores']) for s in samples])
    ):
        mask_np = mask.numpy()
        n_total = len(mask_np)
        n_masked = mask_np.sum()
        ratio = n_masked / n_total

        masked_scores = scores_raw[mask_np]
        unmasked_scores = scores_raw[~mask_np]

        print(f"\nExemplo {i+1}:")
        print(f"  Patches totais     : {n_total}")
        print(f"  Patches mascarados : {n_masked} ({ratio:.1%})")
        print(f"  Score médio mascarados   : {masked_scores.mean():.4f}")
        print(f"  Score médio não-mascarados: {unmasked_scores.mean():.4f}")
        print(f"  (score mascarados > não-mascarados indica seletividade)")
    print("=" * 60)
