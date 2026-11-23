"""
Forest plot do TCC (sem argumentos) -> figura_forest.pdf / figura_forest.png (300 dpi).

Le direto dos JSONs de resultados, via `analysis_final.calcular()` (nenhuma
conta nova aqui; so leitura e desenho).
  (a) Estagio 1 menos controle: delta (media das estrategias - controle) por
      semente, media e IC 95% t por recorte (o['msm']); combinado = meta-analise
      principal de efeitos aleatorios (DL + HKSJ, o['meta_msm']['principal']).
  (b) Seletivas menos aleatoria: media de spectral_energy, onset e combined
      menos random, por semente, media e IC 95% t por recorte (o['sel']);
      combinado = o['meta_sel']['principal'].
O AST com 5 folds (1 semente, unidade = fold) fica fora da figura; esta so na
tabela. A explicacao da figura fica na legenda do LaTeX, nao dentro dela.
Faixa cinza: +/- o DP tipico entre sementes (mediana dos recortes).
Cores: azul/laranja validados para daltonismo; o combinado tambem muda de forma
(losango), entao a identidade nao depende so da cor.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import analysis_final as F

FIGURAS = 'figuras'   # pasta de saida, relativa a raiz do repositorio
AZUL, LARANJA = '#2a78d6', '#eb6834'
TINTA, TINTA2, GRADE = '#1f1f1e', '#6b6a63', '#d9d8d2'
# Recortes da figura, na ordem, com a notacao do artigo (o analysis_final fica intocado).
ROTULO = {'gtzan_ast_fixo': 'GTZAN · AST · divisão fixa', 'gtzan_cnn_fixo': 'GTZAN · CNN · divisão fixa',
          'gtzan_cnn_kfold': 'GTZAN · CNN · 5 folds', 'homburg_ast': 'Homburg · AST',
          'fma_ast': 'FMA-small · AST'}
MENOS = '−'


def br(v, fmt='%+.2f'):
    # numero com virgula decimal e sinal de menos tipografico
    return (fmt % v).replace('-', MENOS).replace('.', ',')


def faixa(l):
    return '%s [%s; %s]' % (br(100 * l['est']), br(100 * l['lo']), br(100 * l['hi']))


def linhas(o, por_recorte, combinado):
    L = [dict(rot=ROTULO[c], est=por_recorte[c]['est'], lo=por_recorte[c]['lo'],
              hi=por_recorte[c]['hi'], comb=False) for c in ROTULO]
    L.append(dict(rot='Combinado', est=combinado['est'], lo=combinado['lo'], hi=combinado['hi'], comb=True))
    return L


def painel(ax, L, titulo, banda, xlim):
    ys = np.array([i + (0.5 if l['comb'] else 0) for i, l in enumerate(L)])   # respiro antes do combinado
    ax.axvspan(-banda, banda, color='#ecebe6', zorder=0, lw=0)
    ax.axvline(0, color=TINTA2, lw=1, zorder=1)
    for l, yy in zip(L, ys):
        cor = LARANJA if l['comb'] else AZUL
        ax.plot([100 * l['lo'], 100 * l['hi']], [yy, yy], color=cor, lw=1.8 if l['comb'] else 1.4,
                solid_capstyle='round', zorder=2)
        if l['comb']:
            ax.plot(100 * l['est'], yy, marker='D', ms=6, color=LARANJA, mec='white', mew=1, zorder=3)
        else:
            ax.plot(100 * l['est'], yy, marker='o', ms=5, color=AZUL, mec='white', mew=1, zorder=3)
        ax.text(xlim[1] + 0.02 * (xlim[1] - xlim[0]), yy, faixa(l),
                va='center', ha='left', fontsize=8, color=TINTA2, clip_on=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([l['rot'] for l in L], fontsize=8, color=TINTA)
    for tl, l in zip(ax.get_yticklabels(), L):
        if l['comb']:
            tl.set_fontweight('bold')
    ax.set_ylim(ys[-1] + 0.7, -0.7)
    ax.set_xlim(*xlim)
    ax.set_title(titulo, loc='left', fontsize=9, color=TINTA, fontweight='bold')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, '%g')))
    ax.tick_params(axis='x', labelsize=8, colors=TINTA2)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRADE, lw=0.6, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRADE)


def main():
    o = F.calcular()
    banda = 100 * float(np.nanmedian([o['dp'][x['chave']] for x in o['recortes']]))
    La = linhas(o, o['msm'], o['meta_msm']['principal'])
    Lb = linhas(o, o['sel'], o['meta_sel']['principal'])
    for nome, L in (('(a) Estagio 1 menos controle', La), ('(b) Seletivas menos aleatoria', Lb)):
        print(nome + ':')
        for l in L:
            print('  %-28s %+.2f [%+.2f; %+.2f]' % (l['rot'], 100 * l['est'], 100 * l['lo'], 100 * l['hi']))
    lim = max(max(abs(l['lo']), abs(l['hi'])) for l in La + Lb) * 100
    lim = np.ceil(lim + 0.5)
    fig, axs = plt.subplots(2, 1, figsize=(4.8, 3.4))
    painel(axs[0], La, '(a) Estágio 1 menos controle (p.p.)', banda, (-lim, lim))
    painel(axs[1], Lb, '(b) Seletivas menos aleatória (p.p.)', banda, (-lim, lim))
    axs[1].set_xlabel('diferença de acurácia (p.p.)', fontsize=8.5, color=TINTA2)
    fig.subplots_adjust(left=0.345, right=0.715, top=0.93, bottom=0.12, hspace=0.45)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGURAS, 'figura_forest.' + ext), dpi=300)
    print('figuras/figura_forest.pdf / .png gravados (banda = %.2f pp, eixo +/- %.0f pp)' % (banda, lim))


if __name__ == '__main__':
    main()
