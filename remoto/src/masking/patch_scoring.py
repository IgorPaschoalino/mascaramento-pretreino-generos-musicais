"""
patch_scoring.py
================
Funções de pontuação por patch de espectrograma Mel.

Adaptado de:
    Belfathi et al. (2025) "Is Selective Masking A Key to Improving Domain
    Adaptation for Masked Language Model?" ICAIL 2025.
    https://github.com/ygorg/legal-masking

No trabalho original, palavras de um corpus textual recebem pesos de
relevância (TF-IDF ou metadiscurso) para guiar o mascaramento no BERT.
Aqui, aplicamos a mesma filosofia ao domínio musical: cada patch 2D de
um espectrograma Mel recebe um peso de relevância baseado em critérios
acústicos — energia espectral (análogo ao TF-IDF) e intensidade de onset
rítmico (análogo ao metadiscurso/gênero) — para guiar o mascaramento
seletivo no SSAST.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Classe base
# ---------------------------------------------------------------------------

class PatchScoringFunction:
    """Classe base para funções de pontuação de patches de espectrograma."""

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normaliza os scores para que somem 1 (distribuição de probabilidade)."""
        total = scores.sum()
        if total == 0:
            return np.ones_like(scores) / len(scores)
        return scores / total

    def score_patches(self, patches: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Calcula um score de relevância para cada patch.

        Args:
            patches: array de shape (n_patches, patch_h, patch_w) com os
                     valores do espectrograma em cada patch.
            normalize: se True, normaliza os scores para soma 1.

        Returns:
            array 1D de shape (n_patches,) com os scores.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1) Pontuação por Energia Espectral (análogo ao TF-IDF)
# ---------------------------------------------------------------------------

class SpectralEnergyScoring(PatchScoringFunction):
    """
    Pontuação baseada na energia RMS de cada patch.

    Motivação: patches com alta energia concentram informação tímbrica
    e harmônica relevante para a identidade do gênero musical. Mascarar
    esses patches força o modelo a inferir estruturas espectrais a partir
    do contexto parcial — análogo a mascarar palavras de alto TF-IDF no
    domínio textual.

    Referência da analogia:
        Belfathi et al. (2025), Seção 3.1 (Topic weighting via Tf-Idf).
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__(epsilon)

    def score_patches(self, patches: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Args:
            patches: (n_patches, patch_h, patch_w) — valores do espectrograma
                     (escala log, já normalizados externamente).
            normalize: normalizar para soma 1.

        Returns:
            scores de shape (n_patches,) baseados na energia RMS de cada patch.
        """
        # Energia RMS por patch: raiz da média dos quadrados dos valores
        # Usamos os valores absolutos pois o espectrograma pode ser normalizado
        # para média zero (valores negativos são válidos na escala log)
        rms = np.sqrt(np.mean(patches ** 2, axis=(1, 2)))
        scores = rms + self.epsilon  # garante score > 0 mesmo em patches silenciosos

        if normalize:
            scores = self.normalize(scores)
        return scores


# ---------------------------------------------------------------------------
# 2) Pontuação por Onset Rítmico (análogo ao metadiscurso/gênero)
# ---------------------------------------------------------------------------

class OnsetScoring(PatchScoringFunction):
    """
    Pontuação baseada na intensidade de onset (ataque) de cada patch.

    Motivação: transientes rítmicos — o momento de início de uma nota ou
    batida percussiva — são marcadores estruturantes do gênero musical,
    assim como marcadores de gênero textual ("ordered", "acquitted") são
    estruturantes do discurso jurídico. Mascarar patches com alta
    intensidade de onset força o modelo a inferir a estrutura rítmica
    mesmo na ausência de pistas explícitas.

    Referência da analogia:
        Belfathi et al. (2025), Seção 3.1 (Genre weighting).

    O onset é estimado pela diferença temporal positiva da energia
    (spectral flux), calculada sobre o eixo temporal de cada patch.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__(epsilon)

    def _spectral_flux(self, patch: np.ndarray) -> float:
        """
        Calcula o spectral flux médio de um patch (eixo temporal = eixo 1).

        spectral_flux[t] = sum_f max(0, S[f,t] - S[f,t-1])

        Apenas diferenças positivas (aumentos de energia) são consideradas,
        pois correspondem a ataques/transientes.

        Args:
            patch: (patch_h, patch_w) — espectrograma de um patch.

        Returns:
            float: intensidade média de onset no patch.
        """
        if patch.shape[1] < 2:
            return self.epsilon
        # Diferença ao longo do eixo temporal (largura do patch)
        diff = np.diff(patch, axis=1)
        # Apenas valores positivos (ataques)
        positive_flux = np.maximum(0, diff)
        return float(positive_flux.mean())

    def score_patches(self, patches: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Args:
            patches: (n_patches, patch_h, patch_w).
            normalize: normalizar para soma 1.

        Returns:
            scores de shape (n_patches,) baseados na intensidade de onset.
        """
        scores = np.array([
            self._spectral_flux(p) + self.epsilon
            for p in patches
        ])

        if normalize:
            scores = self.normalize(scores)
        return scores


# ---------------------------------------------------------------------------
# 3) Pontuação Combinada (energia + onset)
# ---------------------------------------------------------------------------

class CombinedScoring(PatchScoringFunction):
    """
    Combina os scores de energia espectral e onset rítmico com peso α.

    score_final = α * score_energia + (1 - α) * score_onset

    O parâmetro α é o hiperparâmetro central da estratégia MSDM proposta.
    Valores próximos de 1 priorizam patches harmonicamente ricos;
    valores próximos de 0 priorizam patches com forte ataque rítmico.
    O valor ótimo de α é determinado por busca em grade sobre o conjunto
    de validação (ver Seção 3.3 da metodologia).

    Referência da analogia:
        Análogo à combinação de estratégias topic + genre em
        Belfathi et al. (2025).
    """

    def __init__(self, alpha: float = 0.5, epsilon: float = 1e-6):
        """
        Args:
            alpha: peso da componente de energia espectral [0, 1].
                   (1 - alpha) é o peso da componente de onset rítmico.
            epsilon: valor mínimo para evitar scores nulos.
        """
        super().__init__(epsilon)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha deve estar em [0, 1], recebido: {alpha}")
        self.alpha = alpha
        self.spectral = SpectralEnergyScoring(epsilon)
        self.onset = OnsetScoring(epsilon)

    def score_patches(self, patches: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Args:
            patches: (n_patches, patch_h, patch_w).
            normalize: normalizar para soma 1.

        Returns:
            scores combinados de shape (n_patches,).
        """
        # Calculamos cada componente sem normalização individual para
        # combinar em escala bruta, e normalizamos apenas o resultado final
        s_energy = self.spectral.score_patches(patches, normalize=False)
        s_onset = self.onset.score_patches(patches, normalize=False)

        # Normalizar cada componente para [0,1] antes de combinar
        # (garante que escalas diferentes não dominem a combinação)
        def minmax(arr):
            lo, hi = arr.min(), arr.max()
            if hi == lo:
                return np.ones_like(arr)
            return (arr - lo) / (hi - lo)

        s_energy = minmax(s_energy)
        s_onset = minmax(s_onset)

        combined = self.alpha * s_energy + (1 - self.alpha) * s_onset
        combined = combined + self.epsilon  # garante positividade

        if normalize:
            combined = self.normalize(combined)
        return combined


# ---------------------------------------------------------------------------
# Registro de estratégias disponíveis
# ---------------------------------------------------------------------------

patch_scoring_strategies = {
    'spectral_energy': SpectralEnergyScoring,  # análogo ao tfidf
    'onset':           OnsetScoring,            # análogo ao metadiscurso
    'combined':        CombinedScoring,         # combinação ponderada (MSDM)
    'random':          None,                    # baseline: mascaramento aleatório
}
