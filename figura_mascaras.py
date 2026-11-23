"""
Figura das mascaras (sem argumentos) -> figura_mascaras.pdf / figura_mascaras.png (300 dpi).

Mostra as mascaras de 75% das SEIS estrategias sobre o MESMO trecho do GTZAN,
usando exatamente o que o pipeline usa:
  - cache real do split fixo (cache/gtzan-<estrategia>-p16-s42, via
    CACHE_STRATEGY: inverse_energy le os scores de spectral_energy,
    specaugment le o cache do random e ignora os scores);
  - um exemplo do split de TESTE (crop inicial de 10 s, random_crop=False);
  - espectrograma 128x992 remontado dos patches com
    masking/spectrogram_utils.patches_to_spectrogram;
  - indices de masking/patch_mask_collator.select_masked, com
    rng=np.random.default_rng(42) novo para cada estrategia.
Imprime a contagem de patches mascarados por estrategia (deve ser 372) e a
sobreposicao de cada uma com a de energia.
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_from_disk

from masking.patch_mask_collator import select_masked, cache_strategy_of
from masking.spectrogram_utils import patches_to_spectrogram

FIGURAS = 'figuras'          # pasta de saida, relativa a raiz do repositorio
IDX = 0                      # primeiro exemplo do teste (rock)
MASK_RATIO = 0.75
FRAME_S = 0.010              # hop 160 amostras a 16 kHz
ESTRATEGIAS = [('random', 'Aleatória'), ('spectral_energy', 'Energia'),
               ('onset', 'Onset'), ('combined', 'Combinada'),
               ('inverse_energy', 'Energia inversa'), ('specaugment', 'Faixas')]


def carregar(strategy):
    path = 'cache/gtzan-%s-p16-s42' % cache_strategy_of(strategy)
    return load_from_disk(path)['test'].with_format('numpy')[IDX]


def main():
    base = carregar('random')
    grid = tuple(int(g) for g in base['grid_shape'])          # (8, 62)
    n_rows, n_cols = grid
    n = n_rows * n_cols
    patches = torch.as_tensor(np.asarray(base['input_values'], dtype=np.float32))
    P = patches.shape[-1]
    spec = patches_to_spectrogram(patches.unsqueeze(0), grid)[0].numpy()   # (128, 992)
    arquivo = os.path.basename(str(base['audio_path']))

    mascaras, conjuntos = {}, {}
    for strat, _ in ESTRATEGIAS:
        ex = carregar(strat)
        # os patches sao os mesmos em todos os caches; conferir
        assert np.allclose(np.asarray(ex['input_values']), patches.numpy(), atol=1e-5), strat
        assert str(ex['audio_path']) == str(base['audio_path']), strat
        idx = select_masked(strat, np.asarray(ex['patch_scores']), n=n, mask_ratio=MASK_RATIO,
                            rng=np.random.default_rng(42), grid_shape=grid,
                            mask_choice='weighted_random', temperature=0.1)
        m = np.zeros(n, dtype=bool)
        m[np.asarray(idx)] = True
        mascaras[strat] = m.reshape(n_rows, n_cols)
        conjuntos[strat] = set(int(i) for i in idx)

    print('Trecho: split de teste, indice %d, %s (crop inicial), espectrograma %dx%d, grade %dx%d'
          % (IDX, arquivo, spec.shape[0], spec.shape[1], n_rows, n_cols))
    e = conjuntos['spectral_energy']
    for strat, nome in ESTRATEGIAS:
        c = int(mascaras[strat].sum())
        ov = len(conjuntos[strat] & e) / len(e)
        print('  %-16s %-16s mascarados = %d   sobreposicao com Energia = %.3f'
              % (strat, nome, c, ov))
        assert c == int(round(n * MASK_RATIO)) == 372, (strat, c)

    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7})
    fig, axs = plt.subplots(4, 2, figsize=(6.3, 3.2), sharex=True, sharey=True)
    paineis = [('Original', None)] + [(nome, mascaras[s]) for s, nome in ESTRATEGIAS]
    T = spec.shape[1] * FRAME_S
    vmin, vmax = np.percentile(spec, [1, 99.5])
    cmap = matplotlib.colormaps['magma'].copy()
    for k, ax in enumerate(axs.flat):
        r, c = divmod(k, 2)
        if k >= len(paineis):
            ax.axis('off')
            ax.text(0.02, 0.0, 'GTZAN, teste #%d: %s\n(10 s iniciais)\nmáscara de 75%%: 372 de 496\npatches em preto'
                    % (IDX, arquivo), transform=ax.transAxes, va='bottom', ha='left', fontsize=7)
            continue
        nome, m = paineis[k]
        img = spec.copy()
        if m is not None:
            full = np.kron(m, np.ones((P, P), dtype=bool))          # (128, 992)
            img = np.ma.masked_where(full, img)
        cm = cmap.with_extremes(bad='black')
        ax.imshow(img, origin='lower', aspect='auto', cmap=cm, vmin=vmin, vmax=vmax,
                  extent=(0, T, 0, spec.shape[0]), interpolation='nearest')
        if m is not None:
            for x in np.arange(1, n_cols) * P * FRAME_S:
                ax.axvline(x, color='0.55', lw=0.15)
            for y in np.arange(1, n_rows) * P:
                ax.axhline(y, color='0.55', lw=0.15)
        ax.set_title(nome, pad=2)
        ax.set_yticks([0, 64, 127])
        ax.set_xticks([0, 2, 4, 6, 8, T])
        ax.set_xticklabels(['0', '2', '4', '6', '8', '9,92'])
        ultimo_da_coluna = (k + 2 >= len(paineis))
        if c == 0:
            ax.set_ylabel('Mel')
        if ultimo_da_coluna:
            if c == 0:
                ax.set_xlabel('tempo (s)')
            ax.tick_params(labelbottom=True)
        else:
            ax.tick_params(labelbottom=False)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.94, bottom=0.11, hspace=0.55, wspace=0.06)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGURAS, 'figura_mascaras.' + ext), dpi=300)
    print('figuras/figura_mascaras.pdf / .png gravados')


if __name__ == '__main__':
    main()
