"""
analysis_ciclo2.py
==================
Analise do ciclo 2 (estudo comparativo, 15/10): controle + 6 estrategias
(4 originais + inverse_energy + specaugment). Sem argumentos; funciona com
resultados PARCIAIS (usa so as sementes/blocos completos) e gera
`analysis_ciclo2.md`.

Fontes (todas opcionais -- o que nao existir e pulado), dentro de RESULTADOS
(padrao: resultados/; ver a constante abaixo):
  - GTZAN AST, split fixo ........ results_v2.json        (+ models_v2/*/test_preds.json, se houver)
  - GTZAN CNN, split fixo ........ results_cnn.json       (+ models_cnn/*/test_preds.json, se houver)
  - GTZAN k-fold (AST e CNN) ..... results_kfold.json     (predicoes embutidas)
  - Homburg v1 (4 originais+ctl) . remoto_v1_homburg/
  - Remoto v2 (Homburg novas, FMA) remoto_v2/

Testes: Friedman por blocos (sementes, ou semente x fold no k-fold), t pareado
por semente, e McNemar exato exemplo a exemplo (no k-fold, sobre as 1000
musicas de cada semente). As mesmas funcoes de analysis_stats/analisar_remoto.
"""

import os
import io
import re
import sys
import json
import math
import itertools

import numpy as np
from scipy import stats

ORIG = ['random', 'spectral_energy', 'onset', 'combined']
NOVAS = ['inverse_energy', 'specaugment']
ESTR = ORIG + NOVAS
CONDS = ['controle'] + ESTR
SAIDA = 'analysis_ciclo2.md'
# Pasta dos resultados salvos (relativa a raiz do repositorio). Para analisar
# resultados de um re-treino, aponte para outra pasta: TCC_RESULTADOS=<pasta>.
RESULTADOS = os.environ.get('TCC_RESULTADOS', 'resultados')
REMOTO_V2 = os.path.join(RESULTADOS, 'remoto_v2')          # RTX 5090: FMA-small + Homburg (2 novas)
REMOTO_V1 = os.path.join(RESULTADOS, 'remoto_v1_homburg')  # RTX 5060: Homburg (controle + 4 originais)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Estatistica
# ---------------------------------------------------------------------------

def t_pareado(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose((a - b).std(ddof=1), 0):
        return float('nan')
    return float(stats.ttest_rel(a, b).pvalue)


def ic_media(d):
    d = np.asarray(d, float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return (float('nan'), float('nan'))
    return stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))


def mcnemar(y, a, b):
    y, a, b = map(np.asarray, (y, a, b))
    ao, bo = a == y, b == y
    n01, n10 = int(np.sum(~ao & bo)), int(np.sum(ao & ~bo))
    p = 1.0 if n01 + n10 == 0 else float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)
    return n01, n10, p


def fmt_p(p):
    return 'degenerado' if p != p else '%.4f' % p


# ---------------------------------------------------------------------------
# Carga: {semente: {condicao: {'acc', 'f1', 'preds'}}}
# ---------------------------------------------------------------------------

def gtzan_ast():
    r = load(os.path.join(RESULTADOS, 'results_v2.json')) or {}
    blocos = {}
    for k, v in r.items():
        m = re.match(r'^(e2|ctl)\|gtzan\|([a-z_]+?)(-s(\d+))?$', k)
        if not m:
            continue                     # e1, mr15/mr25/mr50 etc. ficam de fora
        kind, cond, _, seed = m.groups()
        seed = int(seed) if seed else 42
        cond = 'controle' if kind == 'ctl' else cond
        suf = '' if seed == 42 else '-s%d' % seed
        tag = ('e2ctl-gtzan-noadapt%s' if kind == 'ctl' else 'e2-gtzan-' + cond + '%s') % suf
        preds = load(os.path.join('models_v2', tag, 'test_preds.json'))
        blocos.setdefault(seed, {})[cond] = {'acc': v['test_accuracy'],
                                             'f1': v.get('test_f1_macro'), 'preds': preds}
    return blocos


def gtzan_cnn():
    r = load(os.path.join(RESULTADOS, 'results_cnn.json')) or {}
    blocos = {}
    for k, v in r.items():
        m = re.match(r'^cnn\|gtzan\|([a-z_]+)-s(\d+)$', k)
        if not m:
            continue
        cond, seed = m.group(1), int(m.group(2))
        cond = 'controle' if cond == 'noadapt' else cond
        tag = 'cnn-gtzan-%s-s%d' % ('noadapt' if cond == 'controle' else cond, seed)
        preds = load(os.path.join('models_cnn', tag, 'test_preds.json'))
        blocos.setdefault(seed, {})[cond] = {'acc': v['test_accuracy'],
                                             'f1': v.get('test_f1_macro'), 'preds': preds}
    return blocos


def remoto(dir_res, dataset):
    r = load(os.path.join(dir_res, 'resultados_remoto.json')) or {}
    blocos = {}
    for v in r.values():
        if v.get('dataset') != dataset:
            continue
        cond = 'controle' if v['condicao'] == 'ctl' else v['estrategia']
        tag = '%s-%s-%s-s%d' % (dataset, v['condicao'], v['estrategia'], v['semente'])
        preds = load(os.path.join(dir_res, 'execucoes', tag, 'test_preds.json'))
        blocos.setdefault(v['semente'], {})[cond] = {'acc': v['test_accuracy'],
                                                     'f1': v.get('test_f1_macro'),
                                                     'preds': preds}
    return blocos


def juntar(*fontes):
    """Junta blocos de fontes diferentes (ex.: Homburg v1 + v2). Uma condicao
    presente nas duas fica com a da primeira."""
    out = {}
    for f in fontes:
        for s, conds in f.items():
            for c, v in conds.items():
                out.setdefault(s, {}).setdefault(c, v)
    return out


def kfold(arch):
    r = load(os.path.join(RESULTADOS, 'results_kfold.json')) or {}
    por = {}
    for v in r.values():
        if v.get('arch') != arch:
            continue
        cond = 'controle' if v['condition'] == 'noadapt' else v['condition']
        por.setdefault((v['seed'], v['fold']), {})[cond] = v
    return por


# ---------------------------------------------------------------------------
# Relatorio de um conjunto de blocos
# ---------------------------------------------------------------------------

def secao(w, titulo, blocos, nota=''):
    w('## %s\n\n' % titulo)
    if nota:
        w(nota + '\n\n')
    if not blocos:
        w('Sem resultados ainda.\n\n')
        return
    presentes = [c for c in CONDS if any(c in b for b in blocos.values())]
    completas = sorted(s for s, b in blocos.items() if all(c in b for c in presentes))
    falt = sorted(set(blocos) - set(completas))
    w('Condicoes: %s. Sementes completas: **%d** %s%s\n\n'
      % (', '.join(presentes), len(completas), completas,
         ('; incompletas (ignoradas): %s' % falt) if falt else ''))
    if not completas:
        return
    acc = {c: np.array([blocos[s][c]['acc'] for s in completas]) for c in presentes}

    w('| condicao | media | dp | min | max |\n|---|---|---|---|---|\n')
    for c in presentes:
        v = acc[c]
        w('| %s | %.4f | %.4f | %.4f | %.4f |\n'
          % (c, v.mean(), v.std(ddof=1) if len(v) > 1 else 0, v.min(), v.max()))
    w('\n')

    estr = [c for c in presentes if c != 'controle']
    if len(completas) >= 3 and len(estr) >= 3:
        grupos = [('todas as %d estrategias' % len(estr), estr)]
        if len(estr) > 4 and all(e in estr for e in ORIG):
            grupos.append(('4 originais', ORIG))
        for rot, g in grupos:
            chi2, p = stats.friedmanchisquare(*[acc[c] for c in g])
            w('- Friedman, %s: chi2 = %.2f, **p = %.4f**\n' % (rot, chi2, p))
        w('\n')

    if 'controle' in presentes:
        w('| estrategia vs controle | delta (pp) | IC 95% | positivo | p (t pareado) |\n'
          '|---|---|---|---|---|\n')
        ds = []
        for c in estr:
            d = acc[c] - acc['controle']
            ds.append(d)
            lo, hi = ic_media(d)
            w('| %s | %+.2f | [%+.2f; %+.2f] | %d/%d | %s |\n'
              % (c, 100 * d.mean(), 100 * lo, 100 * hi, int((d > 0).sum()), len(d),
                 fmt_p(t_pareado(acc[c], acc['controle']))))
        if len(completas) > 1:
            por_semente = np.mean(np.vstack(ds), axis=0)
            w('\nMSM vs controle, media por semente (conservador): **%+.2f pp**, p = %s\n\n'
              % (100 * por_semente.mean(), fmt_p(t_pareado(por_semente, 0 * por_semente))))

    pares = [('inverse_energy', 'spectral_energy'), ('specaugment', 'random'),
             ('inverse_energy', 'random')]
    linhas = []
    for a, b in pares:
        if a in acc and b in acc:
            d = acc[a] - acc[b]
            lo, hi = ic_media(d)
            linhas.append('| %s vs %s | %+.2f | [%+.2f; %+.2f] | %s |\n'
                          % (a, b, 100 * d.mean(), 100 * lo, 100 * hi,
                             fmt_p(t_pareado(acc[a], acc[b]))))
    if linhas:
        w('| contraste das estrategias novas | delta (pp) | IC 95% | p (t pareado) |\n'
          '|---|---|---|---|\n' + ''.join(linhas) + '\n')

    # McNemar (soma por semente: cada semente e um teste independente)
    mc = []
    for c in estr:
        for s in completas:
            pc, pe = blocos[s].get('controle', {}).get('preds'), blocos[s][c].get('preds')
            if pc and pe:
                mc.append((c, s) + mcnemar(pc['y_true'], pc['y_pred'], pe['y_pred']))
    if mc:
        sig = sum(1 for x in mc if x[4] < 0.05)
        w('McNemar estrategia vs controle: %d de %d comparacoes significativas a 5%% '
          '(~%.1f esperadas por acaso).\n\n' % (sig, len(mc), 0.05 * len(mc)))


def secao_kfold(w, arch):
    por = kfold(arch)
    w('## GTZAN k-fold (5 folds) -- %s\n\n' % arch.upper())
    if not por:
        w('Sem resultados ainda.\n\n')
        return
    presentes = [c for c in CONDS if any(c in b for b in por.values())]
    seeds = sorted({s for s, _ in por})
    # Por semente: acuracia sobre os exemplos dos folds COMPLETOS (todas as condicoes).
    w('| semente | folds completos | n teste | ' + ' | '.join(presentes) + ' |\n')
    w('|---|---|---|' + '---|' * len(presentes) + '\n')
    pooled = {}
    for s in seeds:
        fs = sorted(f for (ss, f), b in por.items() if ss == s and all(c in b for c in presentes))
        if not fs:
            continue
        y = np.concatenate([por[(s, f)][presentes[0]]['test_preds']['y_true'] for f in fs])
        pr = {c: np.concatenate([por[(s, f)][c]['test_preds']['y_pred'] for f in fs])
              for c in presentes}
        pooled[s] = (y, pr, fs)
        w('| %d | %s | %d | %s |\n' % (s, fs, len(y),
                                      ' | '.join('%.4f' % (pr[c] == y).mean() for c in presentes)))
    w('\n')
    # Blocos para Friedman: (semente, fold)
    blocos = [b for (s, f), b in sorted(por.items()) if all(c in b for c in presentes)]
    estr = [c for c in presentes if c != 'controle']
    if len(blocos) >= 3 and len(estr) >= 3:
        acc = {c: np.array([b[c]['test_accuracy'] for b in blocos]) for c in presentes}
        chi2, p = stats.friedmanchisquare(*[acc[c] for c in estr])
        w('- Friedman entre %d estrategias, blocos = semente x fold (n = %d): '
          'chi2 = %.2f, **p = %.4f**\n' % (len(estr), len(blocos), chi2, p))
        if 'controle' in acc:
            d = np.mean([acc[c] - acc['controle'] for c in estr], axis=0)
            w('- MSM vs controle (media das estrategias, por bloco): %+.2f pp, p = %s\n'
              % (100 * d.mean(), fmt_p(t_pareado(d, 0 * d))))
        w('\n')
    if pooled and 'controle' in presentes:
        w('McNemar sobre as musicas de cada semente (estrategia vs controle):\n\n')
        w('| semente | estrategia | delta (pp) | so_estrategia | so_controle | p |\n'
          '|---|---|---|---|---|---|\n')
        for s, (y, pr, fs) in pooled.items():
            for c in estr:
                n01, n10, p = mcnemar(y, pr['controle'], pr[c])
                w('| %d | %s | %+.2f | %d | %d | %.4f |\n'
                  % (s, c, 100 * ((pr[c] == y).mean() - (pr['controle'] == y).mean()),
                     n01, n10, p))
        w('\n')


def main():
    buf = io.StringIO()
    w = buf.write
    w('# Analise do ciclo 2 -- estudo comparativo de estrategias de mascaramento\n\n')
    w('Gerado por `analysis_ciclo2.py`. Regra do projeto: nada e estabelecido '
      'antes de replicar com o dobro das sementes.\n\n')
    secao(w, 'GTZAN -- AST (split fixo, teste n = 150)', gtzan_ast())
    secao(w, 'GTZAN -- CNN (split fixo, teste n = 150)', gtzan_cnn())
    secao_kfold(w, 'cnn')
    secao_kfold(w, 'ast')
    hb = juntar(remoto(REMOTO_V1, 'homburg'), remoto(REMOTO_V2, 'homburg'))
    secao(w, 'Homburg -- AST (teste n = 285)', hb,
          'Controle e 4 originais da corrida v1 (01/10 e 05/10); estrategias novas '
          'da v2, mesmo conjunto de teste (verificado por localizar_dados.py compat).')
    secao(w, 'FMA-small -- AST (teste n = 800)', remoto(REMOTO_V2, 'fma'))
    texto = buf.getvalue()
    with open(SAIDA, 'w', encoding='utf-8') as f:
        f.write(texto)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print(texto)


if __name__ == '__main__':
    main()
