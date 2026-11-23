"""
Forest plot do TCC (sem argumentos) -> figura_forest.pdf / figura_forest.png (300 dpi).

Le direto dos JSONs de resultados, via `analysis_final.calcular()`.
  (a) MSM vs controle: delta (media das estrategias - controle) por semente,
      media e IC 95% t por recorte; combinados por efeitos aleatorios (DL + HKSJ).
  (b) Estrategia vs estrategia: em cada recorte, o par das 4 estrategias
      originais com o maior |limite| do IC 95% simultaneo (Bonferroni, 6 pares)
      -- a maior diferenca compativel com os dados; no fim, o mesmo para o
      combinado e o contraste seletivas - random combinado (IC 95%).
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
CURTO = {'random': 'aleatória', 'spectral_energy': 'energia', 'onset': 'onset',
         'combined': 'combinada', 'inverse_energy': 'energia inversa', 'specaugment': 'faixas'}
# Rotulos dos recortes com a notacao do artigo (o analysis_final fica intocado).
ROTULO = {'gtzan_ast_fixo': 'GTZAN · AST · divisão fixa', 'gtzan_cnn_fixo': 'GTZAN · CNN · divisão fixa',
          'gtzan_cnn_kfold': 'GTZAN · CNN · 5 folds', 'gtzan_ast_kfold': 'GTZAN · AST · 5 folds *',
          'homburg_ast': 'Homburg · AST', 'fma_ast': 'FMA-small · AST'}
MENOS = '−'


def br(v, fmt='%+.2f'):
    # numero com virgula decimal e sinal de menos tipografico
    return (fmt % v).replace('-', MENOS).replace('.', ',')


def faixa(l):
    return '%s [%s; %s]' % (br(100 * l['est']), br(100 * l['lo']), br(100 * l['hi']))


def linhas_msm(o):
    L = []
    for x in o['recortes']:
        m = o['msm'][x['chave']]
        fora = x['unidade'] == 'fold'
        rot = ROTULO[x['chave']]
        nota = '%d %s' % (m['n'], 'folds' if fora else 'sementes')
        if x['chave'] == 'homburg_ast':
            nota += ', 4 primeiras estratégias'
        L.append(dict(rot=rot, est=m['est'], lo=m['lo'], hi=m['hi'], nota=nota, tipo='fora' if fora else 'rec'))
    for lab, key, tipo in (('Combinado', 'principal', 'comb'),
                           ('Combinado, CNN com divisão fixa', 'sens_cnn_fixo', 'sens'),
                           ('Combinado, só AST', 'so_ast', 'sens')):
        r = o['meta_msm'][key]
        L.append(dict(rot=lab, est=r['est'], lo=r['lo'], hi=r['hi'],
                      nota='I²=%.0f%%' % (100 * r['I2']), tipo=tipo))
    return L


def linhas_par(o):
    L = []
    for x in o['recortes']:
        b = o['maxpar_orig'][x['chave']]
        fora = x['unidade'] == 'fold'
        L.append(dict(rot=ROTULO[x['chave']], est=b['est'], lo=b['lo'], hi=b['hi'],
                      nota='%s − %s' % (CURTO[b['par'][0]], CURTO[b['par'][1]]),
                      tipo='fora' if fora else 'rec'))
    r = o['meta_pares']['principal']
    t = r['top']
    L.append(dict(rot='Combinado: maior par', est=t['est'], lo=t['lo_sim'], hi=t['hi_sim'],
                  nota='%s − %s, I²=%.0f%%' % (CURTO[t['par'][0]], CURTO[t['par'][1]], 100 * t['I2']),
                  tipo='comb'))
    s = o['meta_sel']['principal']
    L.append(dict(rot='Combinado: seletivas − aleatória', est=s['est'], lo=s['lo'], hi=s['hi'],
                  nota='IC 95%%, I²=%.0f%%' % (100 * s['I2']), tipo='comb'))
    return L


def painel(ax, L, titulo, banda, xlim):
    n = len(L)
    ys = []
    y = 0
    for i, l in enumerate(L):
        if l['tipo'] in ('comb', 'sens') and (i == 0 or L[i - 1]['tipo'] in ('rec', 'fora')):
            y += 0.6                                   # respiro antes dos combinados
        ys.append(y)
        y += 1
    ys = np.array(ys)
    ax.axvspan(-banda, banda, color='#ecebe6', zorder=0, lw=0)
    ax.axvline(0, color=TINTA2, lw=1, zorder=1)
    for l, yy in zip(L, ys):
        comb = l['tipo'] == 'comb'
        cor = LARANJA if l['tipo'] in ('comb', 'sens') else AZUL
        ax.plot([100 * l['lo'], 100 * l['hi']], [yy, yy], color=cor, lw=2.6 if comb else 2,
                solid_capstyle='round', zorder=2)
        if comb:
            ax.plot(100 * l['est'], yy, marker='D', ms=9, color=LARANJA, mec='white', mew=1.5, zorder=3)
        elif l['tipo'] == 'fora':
            ax.plot(100 * l['est'], yy, marker='o', ms=8, mfc='white', mec=AZUL, mew=2, zorder=3)
        elif l['tipo'] == 'sens':
            ax.plot(100 * l['est'], yy, marker='D', ms=7, mfc='white', mec=LARANJA, mew=1.8, zorder=3)
        else:
            ax.plot(100 * l['est'], yy, marker='o', ms=8, color=AZUL, mec='white', mew=1.5, zorder=3)
        ax.text(xlim[1] + 0.02 * (xlim[1] - xlim[0]), yy,
                '%s   %s' % (faixa(l), l['nota']),
                va='center', ha='left', fontsize=7.5, color=TINTA2, clip_on=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([l['rot'] for l in L], fontsize=8.5, color=TINTA)
    for tl, l in zip(ax.get_yticklabels(), L):
        if l['tipo'] == 'comb':
            tl.set_fontweight('bold')
    ax.set_ylim(ys[-1] + 0.8, -0.8)
    ax.set_xlim(*xlim)
    ax.set_title(titulo, loc='left', fontsize=10, color=TINTA, fontweight='bold')
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
    La, Lb = linhas_msm(o), linhas_par(o)
    print('Rotulos dos recortes (original -> figura):')
    for x in o['recortes']:
        print('  %-34r -> %s' % (x['rotulo'], ROTULO[x['chave']]))
    for nome, L in (('(a)', La), ('(b)', Lb)):
        print('Barras %s:' % nome)
        for l in L:
            print('  %-36s %+.2f [%+.2f; %+.2f]   %s' % (l['rot'], 100 * l['est'], 100 * l['lo'],
                                                    100 * l['hi'], l['nota']))
    lim = max(max(abs(l['lo']), abs(l['hi'])) for l in La + Lb) * 100
    lim = np.ceil(lim + 0.5)
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 7.6), gridspec_kw=dict(height_ratios=[len(La) + 0.6, len(Lb) + 0.6]))
    painel(axs[0], La, '(a) Estágio 1 menos controle (p.p.)',
           banda, (-lim, lim))
    painel(axs[1], Lb, '(b) Maior diferença entre pares de estratégias (p.p.)',
           banda, (-lim, lim))
    axs[1].set_xlabel('diferença de acurácia (p.p.)', fontsize=8.5, color=TINTA2)
    fig.text(0.01, 0.005,
             'Barras: IC 95%% (t por semente; combinados: efeitos aleatórios DL com IC de HKSJ).\n'
             'Em (b), por recorte e no maior par combinado: IC 95%% simultâneo (Bonferroni, 6 pares '
             'das quatro primeiras estratégias).\n'
             'Faixa cinza: ±%s p.p., DP típico entre sementes. * AST com 5 folds: 1 semente, '
             'unidade = fold, fora da combinação. Homburg: só pares da mesma máquina.' % br(banda, '%.2f'), fontsize=6.8, color=TINTA2, va='bottom')
    fig.subplots_adjust(left=0.26, right=0.61, top=0.96, bottom=0.12, hspace=0.32)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGURAS, 'figura_forest.' + ext), dpi=300)
    print('figuras/figura_forest.pdf / .png gravados (banda = %.2f pp, eixo +/- %.0f pp)' % (banda, lim))


if __name__ == '__main__':
    main()
