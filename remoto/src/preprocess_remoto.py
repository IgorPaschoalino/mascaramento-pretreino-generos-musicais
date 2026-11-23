"""
preprocess_remoto.py
====================
Pre-processamento unificado de FMA-small e Homburg: audio -> patches de
espectrograma Mel (float32) + as duas colunas de score base.

Uso:
    python preprocess_remoto.py --dataset fma
    python preprocess_remoto.py --dataset homburg

Sem argumentos obrigatorios alem de --dataset. Resumivel: cada split e gravado
em disco assim que termina, entao uma interrupcao perde no maximo o split em
andamento.

Diferencas em relacao aos pre-processadores do GTZAN/FMA antigos, todas de
ENGENHARIA (os numeros produzidos sao os mesmos):

  1. UM cache por dataset, nao um por estrategia. Os patches -- que sao a parte
     enorme -- eram identicos entre as 4 copias; so as colunas de score
     mudavam. Agora as tres colunas de score convivem no mesmo registro e
     `random` nao precisa de coluna. Economia: ~40 GB -> ~5 GB.

  2. Patches em float32 via Array3D, nao listas Python em float64. Alem de
     metade do disco, Array3D e memory-mapped e carrega muito mais rapido.

  3. Crop aleatorio com RNG derivada de (semente, split, indice) em vez de uma
     RNG unica que avanca ao longo do dataset. Com a RNG compartilhada, retomar
     um split pela metade produzia crops DIFERENTES dos que a execucao original
     teria produzido; com a RNG por exemplo, o crop de cada exemplo e funcao
     so da semente e da sua posicao, entao retomar e exatamente equivalente a
     nao ter interrompido.

Parametros do espectrograma: identicos ao GTZAN, de proposito. Qualquer
divergencia aqui invalidaria a comparacao entre datasets.
"""

import os
import io
import sys
import time
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).parent))

from datasets import (
    Dataset, DatasetDict, Features, Array3D, Sequence, Value,
    disable_progress_bars,
)

# A barra de progresso do from_generator imprime uma linha por exemplo quando
# nao ha TTY real (saida redirecionada para log), o que gera log gigante.
disable_progress_bars()

from masking.patch_scoring import (
    SpectralEnergyScoring, OnsetScoring, CombinedScoring,
)
from remoto_common import DATASETS, genres_of, cache_path_of, STORED_ALPHA

# ---------------------------------------------------------------------------
# Parametros do espectrograma -- IDENTICOS ao pipeline do GTZAN
# ---------------------------------------------------------------------------

SAMPLE_RATE  = 16_000
N_MELS       = 128
HOP_LENGTH   = 160
WIN_LENGTH   = 400
SEGMENT_SECS = 10.0
PATCH_SIZE   = 16
FMIN         = 0
FMAX         = 8_000

FMA_REPO     = "rpmon/fma-genre-classification"
HOMBURG_REPO = "mteb/music-genre"

N_FMA_GENRES = len(genres_of('fma'))

log = logging.getLogger('prep')


def expected_grid():
    """Grade de patches implicada pelos parametros acima: (8, 62) = 496 patches."""
    n_frames = 1 + int(SEGMENT_SECS * SAMPLE_RATE) // HOP_LENGTH
    return (N_MELS // PATCH_SIZE, n_frames // PATCH_SIZE)


GRID = expected_grid()
N_PATCHES = GRID[0] * GRID[1]


def features_schema():
    """Schema explicito. Sem ele, o `datasets` inferiria float64 das listas
    Python e o cache dobraria de tamanho."""
    return Features({
        'input_values': Array3D(shape=(N_PATCHES, PATCH_SIZE, PATCH_SIZE), dtype='float32'),
        # As tres colunas de score. float32 nao e escolha: `datasets` rebaixa
        # float64 para float32 na leitura com with_format('numpy') de qualquer
        # forma (verificado na 4.8.5), e era isso que o GTZAN ja consumia.
        # `score_combined` e gravada pronta, no alpha canonico, porque o
        # pipeline antigo arredondava DEPOIS de combinar -- recombinar a partir
        # de componentes ja arredondadas daria ~1e-8 de diferenca. Custa 2 KB.
        'score_energy':   Sequence(Value('float32'), length=N_PATCHES),
        'score_onset':    Sequence(Value('float32'), length=N_PATCHES),
        'score_combined': Sequence(Value('float32'), length=N_PATCHES),
        'label':          Value('int64'),
        'grid_shape':     Sequence(Value('int64'), length=2),
    })


# ---------------------------------------------------------------------------
# Audio -> patches + scores
# ---------------------------------------------------------------------------

def audio_to_mel(waveform, random_crop, rng):
    seg = int(SEGMENT_SECS * SAMPLE_RATE)
    if len(waveform) >= seg:
        start = int(rng.integers(0, len(waveform) - seg + 1)) if random_crop else 0
        waveform = waveform[start: start + seg]
    else:
        waveform = np.pad(waveform, (0, seg - len(waveform)))
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH, fmin=FMIN, fmax=FMAX)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mean, std = log_mel.mean(), log_mel.std()
    if std > 0:
        log_mel = (log_mel - mean) / std
    return log_mel


def spectrogram_to_patches(spec):
    H, W = spec.shape
    H_t = (H // PATCH_SIZE) * PATCH_SIZE
    W_t = (W // PATCH_SIZE) * PATCH_SIZE
    spec = spec[:H_t, :W_t]
    n_rows, n_cols = H_t // PATCH_SIZE, W_t // PATCH_SIZE
    p = spec.reshape(n_rows, PATCH_SIZE, n_cols, PATCH_SIZE)
    p = p.transpose(0, 2, 1, 3).reshape(-1, PATCH_SIZE, PATCH_SIZE)
    return p, (n_rows, n_cols)


_ENERGY = SpectralEnergyScoring()
_ONSET = OnsetScoring()
_COMBINED = CombinedScoring(alpha=STORED_ALPHA)


def make_record(waveform, label, random_crop, rng):
    """Devolve o registro, ou None se o exemplo deve ser descartado."""
    spec = audio_to_mel(waveform, random_crop, rng)
    patches, grid = spectrogram_to_patches(spec)
    if grid != GRID:
        log.warning('grade inesperada %s (esperada %s) - exemplo descartado', grid, GRID)
        return None
    patches = patches.astype(np.float32)
    # normalize=False: scores CRUS, como o pipeline do GTZAN gravava. Cada uma
    # das tres classes e chamada diretamente, para que a coluna gravada seja
    # literalmente a saida do scorer original -- nada reimplementado aqui.
    return {
        'input_values':   patches,
        'score_energy':   _ENERGY.score_patches(patches, normalize=False),
        'score_onset':    _ONSET.score_patches(patches, normalize=False),
        'score_combined': _COMBINED.score_patches(patches, normalize=False),
        'label':          int(label),
        'grid_shape':     list(grid),
    }


# Indices FIXOS por split. Nao usar hash(str): o hash de strings em Python e
# randomizado por processo (PYTHONHASHSEED), o que tornaria os crops
# irreproduziveis entre execucoes -- exatamente o oposto do objetivo.
_SPLIT_ID = {'train': 0, 'validation': 1, 'test': 2}


def example_rng(seed, split_name, index):
    """RNG deterministica por exemplo -- torna a retomada exata (ver docstring
    do modulo)."""
    return np.random.default_rng([seed, _SPLIT_ID[split_name], index])


# ---------------------------------------------------------------------------
# Fontes de dados
# ---------------------------------------------------------------------------

def source_fma(seed):
    """Splits NATIVOS do repositorio (6400 / 800 / 800).

    Devolve {split: (n, getter)} onde getter(i) -> (waveform, label).
    """
    from datasets import load_dataset, Audio

    log.info('Baixando FMA-small de %s (pode levar bastante tempo na 1a vez)...', FMA_REPO)
    raw = load_dataset(FMA_REPO)
    # decode=False evita o torchcodec, que nao carrega no Windows deste projeto.
    raw = raw.cast_column('audio', Audio(decode=False))
    log.info('FMA-small baixado: %s', {k: len(v) for k, v in raw.items()})

    # O repo publica train/validation. A validation (1600) e dividida ao meio em
    # validation/test, exatamente como o pre-processamento antigo fazia -- assim
    # os 800 de teste sao os mesmos exemplos.
    val_split = raw['validation'].train_test_split(test_size=0.5, seed=seed)
    parts = {
        'train':      raw['train'],
        'validation': val_split['train'],
        'test':       val_split['test'],
    }

    def make_getter(hf_split):
        def getter(i):
            ex = hf_split[i]
            label = ex.get('genre')
            if label is None or not (0 <= int(label) < N_FMA_GENRES):
                return None
            info = ex['audio']
            if isinstance(info, dict):
                data = info.get('bytes')
                if data:
                    wav, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE, mono=True)
                    return wav.astype(np.float32), int(label)
                path = info.get('path') or ''
                if path and os.path.exists(path):
                    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                    return wav.astype(np.float32), int(label)
                return None
            if isinstance(info, np.ndarray):
                return info.astype(np.float32), int(label)
            return None
        return getter

    return {name: (len(sp), make_getter(sp)) for name, sp in parts.items()}


def _stratified_split(items, seed, test_size):
    """Divide [(chave, label)] em duas partes preservando a proporcao de classes."""
    rng = np.random.default_rng(seed)
    by_label = {}
    for key, label in items:
        by_label.setdefault(label, []).append(key)
    part_a, part_b = [], []
    for label, keys in sorted(by_label.items()):
        keys = list(keys)
        rng.shuffle(keys)
        n_b = max(1, int(round(len(keys) * test_size)))
        part_b.extend((k, label) for k in keys[:n_b])
        part_a.extend((k, label) for k in keys[n_b:])
    rng.shuffle(part_a)
    rng.shuffle(part_b)
    return part_a, part_b


def source_homburg(seed):
    """Homburg via huggingface_hub direto.

    `datasets.load_dataset()` neste repo (tipo `audiofolder`) quebra com
    OSError [WinError 127] porque o `torchcodec` e acionado na construcao do
    dataset e sua DLL nativa nao carrega. Listamos os arquivos e baixamos um a
    um, decodificando com librosa -- mesmo contorno do preprocessing_homburg.py.

    Divisao 70/15/15, a MESMA proporcao do GTZAN (o repo publica um unico split).
    """
    from huggingface_hub import HfApi, hf_hub_download

    genres = genres_of('homburg')
    genre2id = {g: i for i, g in enumerate(genres)}

    log.info('Listando arquivos de %s ...', HOMBURG_REPO)
    api = HfApi()
    all_files = api.list_repo_files(repo_id=HOMBURG_REPO, repo_type='dataset')
    audio = [f for f in all_files if f.startswith('data/test/') and f.endswith('.mp3')]
    if not audio:
        raise RuntimeError('Nenhum .mp3 encontrado em %s (esperado data/test/<genero>/*.mp3)'
                           % HOMBURG_REPO)

    items = []
    for f in audio:
        genre = f.split('/')[2]
        if genre in genre2id:
            items.append((f, genre2id[genre]))
    log.info('Homburg: %d arquivos em %d generos', len(items), len(genres))

    train_r, val_r = DATASETS['homburg']['split']
    rest_frac = 1.0 - train_r                       # 0.30
    train_items, rest = _stratified_split(items, seed, test_size=rest_frac)
    # metade do resto para validation, metade para test -> 15% / 15%
    val_items, test_items = _stratified_split(rest, seed, test_size=0.5)
    parts = {'train': train_items, 'validation': val_items, 'test': test_items}

    def make_getter(part):
        def getter(i):
            repo_path, label = part[i]
            local = _download_with_retry(hf_hub_download, repo_path)
            if local is None:
                return None
            wav, _ = librosa.load(local, sr=SAMPLE_RATE, mono=True)
            return wav.astype(np.float32), int(label)
        return getter

    return {name: (len(p), make_getter(p)) for name, p in parts.items()}


def _download_with_retry(hf_hub_download, repo_path, max_retries=6):
    """Caminho rapido se ja em cache local; senao baixa com backoff (o HF
    aplica rate limit em milhares de arquivos pequenos)."""
    try:
        return hf_hub_download(repo_id=HOMBURG_REPO, filename=repo_path,
                               repo_type='dataset', local_files_only=True)
    except Exception:
        pass
    delay = 20
    for attempt in range(1, max_retries + 1):
        try:
            return hf_hub_download(repo_id=HOMBURG_REPO, filename=repo_path,
                                   repo_type='dataset')
        except Exception as exc:
            if attempt == max_retries:
                log.warning('Desistindo de %s apos %d tentativas: %s',
                            repo_path, max_retries, exc)
                return None
            log.warning('Falha em %s (tentativa %d/%d): %s. Aguardando %ds...',
                        repo_path, attempt, max_retries, exc, delay)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    return None


SOURCES = {'fma': source_fma, 'homburg': source_homburg}


# ---------------------------------------------------------------------------
# Construcao
# ---------------------------------------------------------------------------

def build(dataset, seed=42, max_examples=None, cache_root=None):
    cache_dir = Path(cache_root or cache_path_of(dataset))
    if max_examples:
        cache_dir = Path(str(cache_dir) + '-ex%d' % max_examples)
    split_paths = {s: cache_dir / s for s in ('train', 'validation', 'test')}

    if all(p.exists() for p in split_paths.values()):
        log.info('Cache completo ja existe em %s - nada a fazer.', cache_dir)
        return DatasetDict({s: Dataset.load_from_disk(str(p))
                            for s, p in split_paths.items()})

    # A fonte so e construida se houver algum split faltando (evita baixar o
    # dataset inteiro de novo so para descobrir que tudo esta em cache).
    log.info('Preparando fonte de dados para %s ...', dataset)
    sources = SOURCES[dataset](seed)

    out = {}
    for split in ('train', 'validation', 'test'):
        path = split_paths[split]
        if path.exists():
            log.info('[%s] ja em cache - pulando.', split)
            out[split] = Dataset.load_from_disk(str(path))
            continue

        n, getter = sources[split]
        if max_examples:
            n = min(n, max_examples)
        random_crop = (split == 'train')
        log.info('[%s] processando %d exemplos (random_crop=%s)...', split, n, random_crop)

        t0 = time.time()

        def gen(n=n, getter=getter, split=split, random_crop=random_crop):
            ok = skipped = 0
            for i in range(n):
                if i and i % 200 == 0:
                    el = time.time() - t0
                    eta = el / i * (n - i) / 60.0
                    log.info('  [%s] %d/%d  ok=%d  descartados=%d  ETA %.0f min',
                             split, i, n, ok, skipped, eta)
                try:
                    got = getter(i)
                    if got is None:
                        skipped += 1
                        continue
                    waveform, label = got
                    rec = make_record(waveform, label, random_crop,
                                      example_rng(seed, split, i))
                    if rec is None:
                        skipped += 1
                        continue
                    ok += 1
                    yield rec
                except Exception as exc:
                    # MP3s do FMA soltam erros de decode mpg123 com frequencia
                    # (dequantization failed / Illegal Audio-MPEG-Header). Sao
                    # descartaveis por exemplo, nao fatais.
                    skipped += 1
                    log.warning('  [%s] exemplo %d descartado: %s', split, i, exc)
            log.info('  [%s] concluido: %d ok, %d descartados', split, ok, skipped)

        ds = Dataset.from_generator(gen, features=features_schema(),
                                    cache_dir=str(cache_dir / '_build'))
        if len(ds) == 0:
            raise RuntimeError('Nenhum exemplo processado com sucesso em %r.' % split)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(path))
        log.info('[%s] %d exemplos gravados em %s (%.1f min)',
                 split, len(ds), path, (time.time() - t0) / 60.0)
        out[split] = ds

    # O diretorio _build e cache intermediario do `datasets`, redundante com os
    # splits ja gravados.
    build_dir = cache_dir / '_build'
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir, ignore_errors=True)
        log.info('Cache intermediario _build removido.')

    meta = {
        'dataset': dataset,
        'seed': seed,
        'grid_shape': list(GRID),
        'n_patches': N_PATCHES,
        'generos': genres_of(dataset),
        'splits': {s: len(d) for s, d in out.items()},
        'parametros_espectrograma': {
            'sample_rate': SAMPLE_RATE, 'n_mels': N_MELS,
            'hop_length': HOP_LENGTH, 'win_length': WIN_LENGTH,
            'segment_secs': SEGMENT_SECS, 'patch_size': PATCH_SIZE,
            'fmin': FMIN, 'fmax': FMAX,
        },
    }
    with open(cache_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return DatasetDict(out)


def main():
    p = argparse.ArgumentParser(description='Pre-processa FMA-small ou Homburg.')
    p.add_argument('--dataset', required=True, choices=sorted(SOURCES))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-examples', type=int, default=None,
                   help='Limita exemplos por split (teste de fumaca).')
    p.add_argument('--cache-root', default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')

    ds = build(args.dataset, seed=args.seed, max_examples=args.max_examples,
               cache_root=args.cache_root)
    log.info('=' * 60)
    log.info('Concluido. Grade de patches: %s (%d patches por exemplo)', GRID, N_PATCHES)
    for split, d in ds.items():
        log.info('  %-11s %5d exemplos', split, len(d))


if __name__ == '__main__':
    main()
