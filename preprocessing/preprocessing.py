"""
preprocessing.py
================
Converte faixas de áudio do GTZAN em patches de espectrograma Mel com
scores de relevância acústica pré-calculados.

Adaptado de:
    Belfathi et al. (2025) "Is Selective Masking A Key to Improving Domain
    Adaptation for Masked Language Model?" ICAIL 2025.
    https://github.com/ygorg/legal-masking  (preprocessing_dataset.py)

Correspondência com o pipeline original:
    tokenize_function()      → audio_to_mel_spectrogram()
    group_texts()            → spectrogram_to_patches()
    scoring.score_sequence() → scorer.score_patches()
    importance_weight        → patch_scores
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import librosa
from datasets import Dataset, DatasetDict

from masking.patch_scoring import (
    SpectralEnergyScoring,
    OnsetScoring,
    CombinedScoring,
    patch_scoring_strategies,
)


# ---------------------------------------------------------------------------
# Parâmetros padrão de pré-processamento (compatíveis com o SSAST)
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 16_000     # Hz — taxa de amostragem alvo
DEFAULT_N_MELS = 128             # número de bandas Mel
DEFAULT_HOP_LENGTH = 160         # salto em amostras (10 ms a 16kHz)
DEFAULT_WIN_LENGTH = 400         # janela em amostras (25 ms a 16kHz)
DEFAULT_SEGMENT_DURATION = 10.0  # segundos — duração do segmento de áudio
DEFAULT_PATCH_SIZE = 16          # tamanho do patch quadrado (pixels)
DEFAULT_FMIN = 0                 # frequência mínima do banco Mel
DEFAULT_FMAX = 8_000             # frequência máxima do banco Mel


# ---------------------------------------------------------------------------
# 1) Áudio → Espectrograma Mel
# ---------------------------------------------------------------------------

def audio_to_mel_spectrogram(
    audio_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
    hop_length: int = DEFAULT_HOP_LENGTH,
    win_length: int = DEFAULT_WIN_LENGTH,
    segment_duration: float = DEFAULT_SEGMENT_DURATION,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    random_crop: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Carrega um arquivo de áudio e retorna um espectrograma Mel 2D.

    Análogo a tokenize_function() de Belfathi et al. (2025): assim como
    aquela função converte texto em tokens, esta converte áudio em sua
    representação discreta (espectrograma Mel).

    Args:
        audio_path: caminho para o arquivo de áudio (.wav, .mp3, etc.).
        sample_rate: taxa de amostragem alvo (reamostrado se necessário).
        n_mels: número de filtros Mel.
        hop_length: salto em amostras entre frames consecutivos.
        win_length: tamanho da janela STFT em amostras.
        segment_duration: duração do segmento extraído (em segundos).
        fmin, fmax: limites do banco de filtros Mel.
        random_crop: se True, extrai segmento aleatório; senão, do início.
        rng: gerador numpy para reprodutibilidade do crop aleatório.

    Returns:
        Espectrograma Mel de shape (n_mels, T) em escala logarítmica,
        normalizado para média 0 e desvio padrão 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Carregar e reamostrar
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Extrair segmento de comprimento fixo
    segment_samples = int(segment_duration * sample_rate)
    if len(waveform) >= segment_samples:
        if random_crop:
            max_start = len(waveform) - segment_samples
            start = rng.integers(0, max_start + 1)
        else:
            start = 0
        waveform = waveform[start: start + segment_samples]
    else:
        # Padding com zeros se o áudio for menor que o segmento
        pad = segment_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad), mode='constant')

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
    )

    # Escala logarítmica (dB)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalização z-score (por espectrograma individual)
    # A normalização global pelo conjunto de treino é aplicada em normalize_dataset()
    mean = log_mel.mean()
    std = log_mel.std()
    if std > 0:
        log_mel = (log_mel - mean) / std

    return log_mel  # shape: (n_mels, T)


# ---------------------------------------------------------------------------
# 2) Espectrograma → Patches
# ---------------------------------------------------------------------------

def spectrogram_to_patches(
    spectrogram: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Divide um espectrograma 2D em patches quadrados não sobrepostos.

    Análogo a group_texts() de Belfathi et al. (2025): assim como aquela
    função divide o texto tokenizado em chunks de tamanho fixo, esta divide
    o espectrograma em patches de tamanho fixo.

    Args:
        spectrogram: array (n_mels, T).
        patch_size: lado do patch quadrado em pixels.

    Returns:
        patches: array (n_patches, patch_size, patch_size).
        grid_shape: tupla (n_rows, n_cols) descrevendo a grade de patches.
    """
    H, W = spectrogram.shape

    # Truncar para múltiplo de patch_size em ambas as dimensões
    H_trunc = (H // patch_size) * patch_size
    W_trunc = (W // patch_size) * patch_size
    spec = spectrogram[:H_trunc, :W_trunc]

    n_rows = H_trunc // patch_size
    n_cols = W_trunc // patch_size

    # Reshape: (H, W) → (n_rows, patch_h, n_cols, patch_w) → (n_patches, pH, pW)
    patches = spec.reshape(n_rows, patch_size, n_cols, patch_size)
    patches = patches.transpose(0, 2, 1, 3)          # (n_rows, n_cols, pH, pW)
    patches = patches.reshape(-1, patch_size, patch_size)  # (n_patches, pH, pW)

    return patches, (n_rows, n_cols)


# ---------------------------------------------------------------------------
# 3) Pré-processamento de um único arquivo
# ---------------------------------------------------------------------------

def preprocess_audio_file(
    audio_path: str,
    label: int,
    scorer,
    patch_size: int = DEFAULT_PATCH_SIZE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
    segment_duration: float = DEFAULT_SEGMENT_DURATION,
    random_crop: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Pipeline completo para um arquivo: áudio → espectrograma → patches → scores.

    Análogo à sequência tokenize → reconstruct_words → score_sequence do
    preprocessing_dataset.py de Belfathi et al. (2025), executada arquivo a arquivo.

    Args:
        audio_path: caminho do arquivo de áudio.
        label: índice do gênero musical (0–9 para GTZAN).
        scorer: instância de PatchScoringFunction (ou None para random baseline).
        patch_size: lado dos patches em pixels.
        sample_rate, n_mels, segment_duration: parâmetros do espectrograma.
        random_crop: crop aleatório no eixo temporal.
        rng: gerador numpy.

    Returns:
        Dicionário com:
            'input_values'  : list[list[list[float]]] — patches (n_patches, pH, pW)
            'patch_scores'  : list[float] — scores por patch
            'label'         : int — rótulo do gênero
            'grid_shape'    : list[int] — [n_rows, n_cols]
            'audio_path'    : str — caminho original
    """
    if rng is None:
        rng = np.random.default_rng()

    # Passo 1: áudio → espectrograma
    spectrogram = audio_to_mel_spectrogram(
        audio_path,
        sample_rate=sample_rate,
        n_mels=n_mels,
        segment_duration=segment_duration,
        random_crop=random_crop,
        rng=rng,
    )

    # Passo 2: espectrograma → patches
    patches, grid_shape = spectrogram_to_patches(spectrogram, patch_size)

    # Passo 3: calcular scores de relevância por patch
    if scorer is not None:
        scores = scorer.score_patches(patches, normalize=False)
    else:
        # Baseline aleatório: todos os patches têm peso igual
        scores = np.ones(len(patches))

    return {
        'input_values': patches.tolist(),
        'patch_scores': scores.tolist(),
        'label': label,
        'grid_shape': list(grid_shape),
        'audio_path': audio_path,
    }


# ---------------------------------------------------------------------------
# 4) Pré-processamento do GTZAN completo
# ---------------------------------------------------------------------------

GTZAN_GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]


def load_gtzan(data_dir: str):
    """
    Carrega os caminhos e rótulos do GTZAN a partir de uma estrutura de diretórios.

    Estrutura esperada:
        data_dir/
          genres/
            blues/       (100 arquivos .wav)
            classical/
            ...

    Args:
        data_dir: raiz do dataset GTZAN.

    Returns:
        Lista de tuplas (audio_path, label_idx).
    """
    genres_dir = Path(data_dir) / 'genres'
    if not genres_dir.exists():
        raise FileNotFoundError(
            f"Diretório de gêneros não encontrado: {genres_dir}\n"
            f"Baixe o GTZAN em: https://marsyas.info/downloads/datasets.html"
        )

    samples = []
    for label_idx, genre in enumerate(GTZAN_GENRES):
        genre_dir = genres_dir / genre
        if not genre_dir.exists():
            logging.warning(f"Diretório não encontrado para gênero: {genre}")
            continue
        for wav_file in sorted(genre_dir.glob('*.wav')):
            samples.append((str(wav_file), label_idx))

    logging.info(f"GTZAN: {len(samples)} arquivos encontrados em {len(GTZAN_GENRES)} gêneros.")
    return samples


def build_dataset(
    data_dir: str,
    mask_strategy: str = 'combined',
    patch_size: int = DEFAULT_PATCH_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    alpha: float = 0.5,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> DatasetDict:
    """
    Constrói o DatasetDict completo (train/validation/test) com patches e scores.

    Análogo ao main() do preprocessing_dataset.py de Belfathi et al. (2025):
    carrega o corpus, instancia a função de scoring, aplica o pré-processamento
    e salva em cache.

    Args:
        data_dir: raiz do GTZAN.
        mask_strategy: estratégia de scoring ('spectral_energy', 'onset',
                       'combined', 'random').
        patch_size: tamanho dos patches.
        train_ratio, val_ratio: proporções das divisões.
        alpha: parâmetro de combinação para CombinedScoring.
        cache_dir: diretório para cache em disco (None = sem cache).
        seed: semente para reprodutibilidade.

    Returns:
        DatasetDict com splits 'train', 'validation', 'test'.
    """
    rng = np.random.default_rng(seed)

    # Verificar cache
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"gtzan-{mask_strategy}-p{patch_size}-s{seed}"
        if cache_path.exists():
            logging.info(f"Carregando dataset do cache: {cache_path}")
            return DatasetDict.load_from_disk(str(cache_path))

    # Carregar caminhos do GTZAN
    all_samples = load_gtzan(data_dir)

    # Embaralhar e dividir em splits estratificados por gênero
    rng.shuffle(all_samples)  # type: ignore
    train_data, val_data, test_data = stratified_split(
        all_samples, train_ratio, val_ratio, seed
    )
    logging.info(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Instanciar função de scoring
    if mask_strategy == 'spectral_energy':
        scorer = SpectralEnergyScoring()
    elif mask_strategy == 'onset':
        scorer = OnsetScoring()
    elif mask_strategy == 'combined':
        scorer = CombinedScoring(alpha=alpha)
    elif mask_strategy == 'random':
        scorer = None
    else:
        raise ValueError(f"Estratégia desconhecida: {mask_strategy}. "
                         f"Opções: {list(patch_scoring_strategies.keys())}")

    # Pré-processar cada split
    def process_split(samples, split_name, random_crop=True):
        records = []
        for i, (path, label) in enumerate(samples):
            if i % 100 == 0:
                logging.info(f"  [{split_name}] Processando {i}/{len(samples)}: {path}")
            try:
                record = preprocess_audio_file(
                    path, label, scorer,
                    patch_size=patch_size,
                    random_crop=random_crop,
                    rng=rng,
                )
                records.append(record)
            except Exception as e:
                logging.warning(f"Erro ao processar {path}: {e}")
        return records

    logging.info("Pré-processando split de treino...")
    train_records = process_split(train_data, 'train', random_crop=True)

    logging.info("Pré-processando split de validação...")
    val_records = process_split(val_data, 'validation', random_crop=False)

    logging.info("Pré-processando split de teste...")
    test_records = process_split(test_data, 'test', random_crop=False)

    # Converter para DatasetDict
    def records_to_dataset(records):
        if not records:
            raise RuntimeError("Nenhum registro foi processado com sucesso.")
        keys = records[0].keys()
        return Dataset.from_dict({k: [r[k] for r in records] for k in keys})

    dataset_dict = DatasetDict({
        'train': records_to_dataset(train_records),
        'validation': records_to_dataset(val_records),
        'test': records_to_dataset(test_records),
    })

    # Salvar cache
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        logging.info(f"Salvando cache em {cache_path}")
        dataset_dict.save_to_disk(str(cache_path))

    return dataset_dict


def stratified_split(samples, train_ratio, val_ratio, seed):
    """
    Divide os samples mantendo a proporção de classes em cada split.

    Args:
        samples: lista de (path, label).
        train_ratio, val_ratio: proporções (test_ratio = 1 - train - val).
        seed: semente.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)

    train, val, test = [], [], []
    for label, class_samples in by_class.items():
        arr = np.array(class_samples, dtype=object)
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(arr[:n_train].tolist())
        val.extend(arr[n_train:n_train + n_val].tolist())
        test.extend(arr[n_train + n_val:].tolist())

    return train, val, test


# ---------------------------------------------------------------------------
# CLI (análogo ao main() do preprocessing_dataset.py deles)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pré-processa o GTZAN: áudio → patches de espectrograma com scores."
    )
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Raiz do dataset GTZAN.')
    parser.add_argument('--mask-strategy', type=str,
                        choices=list(patch_scoring_strategies.keys()),
                        default='combined',
                        help='Estratégia de scoring para os patches.')
    parser.add_argument('--patch-size', type=int, default=DEFAULT_PATCH_SIZE,
                        help=f'Tamanho do patch (padrão: {DEFAULT_PATCH_SIZE}).')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Peso da energia espectral na estratégia combined [0,1].')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Diretório de cache (padrão: ./cache).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semente aleatória (padrão: 42).')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    logging.info(f"Estratégia de mascaramento : {args.mask_strategy}")
    logging.info(f"Tamanho do patch           : {args.patch_size}")
    logging.info(f"Alpha (combined)           : {args.alpha}")
    logging.info(f"Cache                      : {args.cache_dir}")

    dataset = build_dataset(
        data_dir=args.data_dir,
        mask_strategy=args.mask_strategy,
        patch_size=args.patch_size,
        alpha=args.alpha,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    logging.info("Pré-processamento concluído.")
    for split, ds in dataset.items():
        logging.info(f"  {split}: {len(ds)} exemplos")


if __name__ == '__main__':
    main()
