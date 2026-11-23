"""
Analise final do TCC (sem argumentos) -- so dados existentes, nenhum treino.

Gera `analysis_final.md` com:
  - Homburg separado por maquina (v1 RTX 5060: controle + 4 originais;
    v2 RTX 5090: inverse_energy + specaugment);
  - tabela de todos os testes de hipotese do protocolo final, com Holm em duas
    familias: (a) estrategia vs estrategia, (b) MSM vs controle;
  - meta-analise de efeitos aleatorios (DerSimonian-Laird + IC de
    Hartung-Knapp-Sidik-Jonkman) para (a) MSM vs controle e (b) a maior
    diferenca entre estrategias compativel com os dados;
  - frases "efeitos maiores que X pp estao excluidos" e o desvio-padrao entre
    sementes de cada recorte (criterio de relevancia pratica).

Unidade de analise: a SEMENTE. Em cada recorte, o delta e pareado dentro da
semente. No k-fold, os 5 folds de uma semente sao agregados (acuracia sobre as
999 musicas) e a semente vira uma unidade; os folds nao sao replicas
independentes (compartilham ~75% dos dados de treino entre si e a mesma
particao em todas as sementes). O AST k-fold tem uma semente so: entra nas
tabelas por recorte com os folds como unidade (marcado), mas fica FORA das
meta-analises, porque nao estima a variancia entre sementes.

Le os mesmos JSONs que `analysis_ciclo2.py` (reaproveita as funcoes de carga).
`figura_forest.py` importa `calcular()` daqui.
"""
import io
import itertools
import sys

import numpy as np
from scipy import stats

import analysis_ciclo2 as A

ORIG, NOVAS, ESTR = A.ORIG, A.NOVAS, A.ESTR
SAIDA = 'analysis_final.md'

# Homburg: grupos de maquina. Contrastes entre grupos ficam confundidos.
HB_V1 = ['controle'] + ORIG
HB_V2 = NOVAS


# ---------------------------------------------------------------------------
# Carga: cada recorte vira {unidade: {condicao: acuracia}}
# ---------------------------------------------------------------------------

def _fixo(blocos):
    return {s: {c: v['acc'] for c, v in b.items()} for s, b in blocos.items()}


def _kfold_por_semente(arch):
    por = A.kfold(arch)
    conds = [c for c in A.CONDS if any(c in b for b in por.values())]
    out = {}
    for s in sorted({s for s, _ in por}):
        fs = [f for (ss, f), b in por.items() if ss == s and all(c in b for c in conds)]
        if len(fs) < 5:
            continue
        y = np.concatenate([por[(s, f)][conds[0]]['test_preds']['y_true'] for f in sorted(fs)])
        out[s] = {}
        for c in conds:
            p = np.concatenate([por[(s, f)][c]['test_preds']['y_pred'] for f in sorted(fs)])
            out[s][c] = float((p == y).mean())
    return out


def _kfold_por_fold(arch, seed=42):
    por = A.kfold(arch)
    return {f: {c: v['test_accuracy'] for c, v in b.items()}
            for (s, f), b in sorted(por.items()) if s == seed}


def recortes():
    """Lista de recortes: dict(chave, rotulo, unidade, dados, conds, meta)."""
    hb = _fixo(A.juntar(A.remoto(A.REMOTO_V1, 'homburg'), A.remoto(A.REMOTO_V2, 'homburg')))
    r = [
        dict(chave='gtzan_ast_fixo', rotulo='GTZAN · AST · fixo', unidade='semente',
             dados=_fixo(A.gtzan_ast())),
        dict(chave='gtzan_cnn_fixo', rotulo='GTZAN · CNN · fixo', unidade='semente',
             dados=_fixo(A.gtzan_cnn())),
        dict(chave='gtzan_cnn_kfold', rotulo='GTZAN · CNN · k-fold', unidade='semente',
             dados=_kfold_por_semente('cnn')),
        dict(chave='gtzan_ast_kfold', rotulo='GTZAN · AST · k-fold (1 sem.)', unidade='fold',
             dados=_kfold_por_fold('ast')),
        dict(chave='homburg_ast', rotulo='Homburg · AST · fixo', unidade='semente', dados=hb),
        dict(chave='fma_ast', rotulo='FMA-small · AST · fixo', unidade='semente',
             dados=_fixo(A.remoto(A.REMOTO_V2, 'fma'))),
    ]
    for x in r:
        conds = [c for c in A.CONDS if all(c in b for b in x['dados'].values())]
        x['conds'] = conds
        x['estr'] = [c for c in conds if c != 'controle']
        x['n'] = len(x['dados'])
        # Estrategias usadas no MSM vs controle e pares sem confusao de maquina
        if x['chave'] == 'homburg_ast':
            x['estr_msm'] = ORIG
            x['pares'] = list(itertools.combinations(ORIG, 2)) + [tuple(NOVAS)]
            x['friedman'] = ORIG
        else:
            x['estr_msm'] = x['estr']
            x['pares'] = list(itertools.combinations(x['estr'], 2))
            x['friedman'] = x['estr']
    return r


# Meta-analise principal: um recorte por (base x arquitetura), sem reusar dados.
# CNN GTZAN: k-fold (999 musicas de teste por semente, contra 150 do fixo) --
# regra fixada antes de olhar o resultado combinado: fica o recorte de maior
# conjunto de teste. O fixo entra na analise de sensibilidade.
META_PRINCIPAL = ['gtzan_ast_fixo', 'gtzan_cnn_kfold', 'homburg_ast', 'fma_ast']
META_SENS_CNN_FIXO = ['gtzan_ast_fixo', 'gtzan_cnn_fixo', 'homburg_ast', 'fma_ast']
META_SO_AST = ['gtzan_ast_fixo', 'homburg_ast', 'fma_ast']


# ---------------------------------------------------------------------------
# Estatistica
# ---------------------------------------------------------------------------

def resumo(d, conf=0.95):
    """Media, EP, gl, IC t e p (t contra zero) de deltas pareados."""
    d = np.asarray(d, float)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
    p = float(stats.ttest_1samp(d, 0).pvalue) if se > 0 else float('nan')
    return dict(est=m, se=se, df=n - 1, lo=m - q * se, hi=m + q * se, p=p, n=n,
                pos=int((d > 0).sum()))


def delta_msm(x):
    """Por unidade: media das estrategias (estr_msm) menos o controle."""
    u = sorted(x['dados'])
    return np.array([np.mean([x['dados'][s][c] for c in x['estr_msm']])
                     - x['dados'][s]['controle'] for s in u])


def delta_par(x, a, b):
    u = sorted(x['dados'])
    return np.array([x['dados'][s][a] - x['dados'][s][b] for s in u])


def friedman(x, conds):
    u = sorted(x['dados'])
    chi, p = stats.friedmanchisquare(*[[x['dados'][s][c] for s in u] for c in conds])
    return float(chi), float(p)


def dp_sementes(x):
    """DP entre sementes da acuracia, combinado entre condicoes (raiz da media
    das variancias). E o 'ruido de trocar a semente' do recorte."""
    u = sorted(x['dados'])
    v = [np.var([x['dados'][s][c] for s in u], ddof=1) for c in x['conds']]
    return float(np.sqrt(np.mean(v)))


def meta_dl(est, se, conf=0.95):
    """DerSimonian-Laird; IC normal e IC de Hartung-Knapp-Sidik-Jonkman (t com
    k-1 gl, com o fator q truncado em 1 para nunca ficar mais estreito que o DL)."""
    y, v = np.asarray(est, float), np.asarray(se, float) ** 2
    k = len(y)
    w = 1 / v
    mu_f = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - mu_f) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c)
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    se_dl = float(np.sqrt(1 / np.sum(ws)))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    qhk = float(np.sum(ws * (y - mu) ** 2) / (k - 1))
    se_hk = float(np.sqrt(max(qhk, 1.0) / np.sum(ws)))
    t = stats.t.ppf(1 - (1 - conf) / 2, k - 1)
    return dict(est=mu, k=k, tau2=tau2, tau=np.sqrt(tau2), I2=I2, Q=Q,
                pQ=float(stats.chi2.sf(Q, k - 1)),
                lo_dl=mu - z * se_dl, hi_dl=mu + z * se_dl,
                p_dl=float(2 * stats.norm.sf(abs(mu / se_dl))),
                lo=mu - t * se_hk, hi=mu + t * se_hk,
                p=float(2 * stats.t.sf(abs(mu / se_hk), k - 1)))


def holm(ps):
    ps = np.asarray(ps, float)
    m = len(ps)
    ordem = np.argsort(ps)
    adj = np.empty(m)
    corr = 0.0
    for i, j in enumerate(ordem):
        corr = max(corr, min(1.0, (m - i) * ps[j]))
        adj[j] = corr
    return adj


# ---------------------------------------------------------------------------
# Calculo central (usado tambem por figura_forest.py)
# ---------------------------------------------------------------------------

def calcular():
    R = recortes()
    por = {x['chave']: x for x in R}
    out = dict(recortes=R, msm={}, pares={}, maxpar={}, friedman={}, dp={})
    for x in R:
        out['dp'][x['chave']] = dp_sementes(x) if x['unidade'] == 'semente' else float('nan')
        out['msm'][x['chave']] = resumo(delta_msm(x))
        out['friedman'][x['chave']] = (len(x['friedman']), x['n']) + friedman(x, x['friedman'])
        m = len(x['pares'])
        lst = []
        for a, b in x['pares']:
            r = resumo(delta_par(x, a, b))
            rs = resumo(delta_par(x, a, b), conf=1 - 0.05 / m)   # Bonferroni simultaneo
            r.update(lo_sim=rs['lo'], hi_sim=rs['hi'], a=a, b=b)
            lst.append(r)
        out['pares'][x['chave']] = lst
        # Maior diferenca compativel: o maior |limite| dos ICs simultaneos.
        top = max(lst, key=lambda r: max(abs(r['lo_sim']), abs(r['hi_sim'])))
        out['maxpar'][x['chave']] = dict(par=(top['a'], top['b']), est=top['est'],
                                         lo=top['lo_sim'], hi=top['hi_sim'],
                                         X=max(abs(top['lo_sim']), abs(top['hi_sim'])), m=m)

    def meta_msm(chaves):
        r = meta_dl([out['msm'][c]['est'] for c in chaves], [out['msm'][c]['se'] for c in chaves])
        r['recortes'] = chaves
        return r
    out['meta_msm'] = {'principal': meta_msm(META_PRINCIPAL),
                       'sens_cnn_fixo': meta_msm(META_SENS_CNN_FIXO),
                       'so_ast': meta_msm(META_SO_AST)}

    def meta_pares(chaves, conj):
        """Combina cada par de `conj` entre os recortes que o tem. X = maior
        |limite| dos ICs simultaneos (Bonferroni sobre os m pares)."""
        todos = sorted({tuple(sorted((r['a'], r['b']))) for c in chaves for r in out['pares'][c]
                        if r['a'] in conj and r['b'] in conj})
        res = []
        for par in todos:
            ys, ses, usados = [], [], []
            for c in chaves:
                for r in out['pares'][c]:
                    if tuple(sorted((r['a'], r['b']))) == par:
                        sgn = 1 if (r['a'], r['b']) == par else -1
                        ys.append(sgn * r['est'])
                        ses.append(r['se'])
                        usados.append(c)
            if len(ys) < 2:
                continue
            mr = meta_dl(ys, ses)
            mr.update(par=par, recortes=usados)
            res.append(mr)
        m = len(res)
        for mr in res:
            k = mr['k']
            se_hk = (mr['hi'] - mr['est']) / stats.t.ppf(0.975, k - 1)
            q = stats.t.ppf(1 - 0.025 / m, k - 1)
            mr['lo_sim'], mr['hi_sim'] = mr['est'] - q * se_hk, mr['est'] + q * se_hk
        top = max(res, key=lambda r: max(abs(r['lo_sim']), abs(r['hi_sim'])))
        maior = max(res, key=lambda r: abs(r['est']))
        return dict(pares=res, m=m, top=top, maior=maior, kmin=min(r['k'] for r in res),
                    X=max(abs(top['lo_sim']), abs(top['hi_sim'])))
    # Medida principal de (b): os 6 pares das 4 estrategias originais, presentes
    # nos 4 recortes da analise principal (k = 4 em todos). Pares com as novas
    # tem k = 2-3 -- com HKSJ + Bonferroni o IC fica sem sentido (t com 1-2 gl);
    # eles aparecem so na tabela por par.
    out['meta_pares'] = {'principal': meta_pares(META_PRINCIPAL, ORIG),
                         'sens_cnn_fixo': meta_pares(META_SENS_CNN_FIXO, ORIG),
                         'so_ast': meta_pares(META_SO_AST, ORIG),
                         'todos_pares': meta_pares(META_PRINCIPAL, ESTR)}

    # Suplementar: seletivas (spectral_energy, onset, combined) menos random --
    # a pergunta original do TCC num unico contraste por semente.
    SEL = ['spectral_energy', 'onset', 'combined']
    for x in R:
        u = sorted(x['dados'])
        d = [np.mean([x['dados'][s][c] for c in SEL]) - x['dados'][s]['random'] for s in u]
        out.setdefault('sel', {})[x['chave']] = resumo(d)

    def meta_sel(chaves):
        r = meta_dl([out['sel'][c]['est'] for c in chaves], [out['sel'][c]['se'] for c in chaves])
        r['recortes'] = chaves
        return r
    out['meta_sel'] = {'principal': meta_sel(META_PRINCIPAL),
                       'sens_cnn_fixo': meta_sel(META_SENS_CNN_FIXO),
                       'so_ast': meta_sel(META_SO_AST)}
    # X por recorte restrito aos 6 pares originais (comparavel ao combinado)
    for x in R:
        lst = [r for r in out['pares'][x['chave']] if r['a'] in ORIG and r['b'] in ORIG]
        m = len(lst)
        best = None
        for r in lst:
            rs = resumo(delta_par(x, r['a'], r['b']), conf=1 - 0.05 / m)
            X = max(abs(rs['lo']), abs(rs['hi']))
            if best is None or X > best['X']:
                best = dict(par=(r['a'], r['b']), est=r['est'], lo=rs['lo'], hi=rs['hi'], X=X, m=m)
        out['maxpar_orig'] = out.get('maxpar_orig', {})
        out['maxpar_orig'][x['chave']] = best
    out['por'] = por
    return out


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def pp(x):
    return '%+.2f' % (100 * x)


def fp(p):
    return '<0.0001' if p < 1e-4 else '%.4f' % p


def main():
    o = calcular()
    R, por = o['recortes'], o['por']
    buf = io.StringIO()
    w = buf.write
    w('# Analise final -- Homburg por maquina, Holm, meta-analise, limites de efeito\n\n')
    w('Gerado por `analysis_final.py` (sem argumentos; le os JSONs de resultados). '
      'Unidade: semente (k-fold: semente com os 5 folds agregados, 999 musicas). '
      'Deltas em pontos percentuais (pp). Nada aqui envolveu treino novo.\n\n')

    # ---- recortes
    w('## Recortes\n\n| recorte | unidade | n | condicoes | DP entre sementes (pp) |\n'
      '|---|---|---|---|---|\n')
    for x in R:
        d = o['dp'][x['chave']]
        w('| %s | %s | %d | %d | %s |\n' % (x['rotulo'], x['unidade'], x['n'], len(x['conds']),
                                           'n/a (1 semente)' if d != d else '%.2f' % (100 * d)))
    w('\nDP entre sementes = raiz da media, entre condicoes, da variancia da acuracia entre '
      'sementes. E o "ruido de trocar a semente": um efeito menor que ele nao tem valor pratico.\n\n')

    # ---- Item 2: Homburg
    hb = por['homburg_ast']
    w('## Homburg separado por maquina\n\n')
    w('v1 (RTX 5060): controle + random, spectral_energy, onset, combined. '
      'v2 (RTX 5090): inverse_energy, specaugment. Mesmo conjunto de teste (285).\n\n')
    k, n, chi, p = o['friedman']['homburg_ast']
    w('- **Friedman, 4 originais (v1, mesma maquina):** chi2 = %.2f, p = %s (n = %d sementes)\n'
      % (chi, fp(p), n))
    rv2 = [r for r in o['pares']['homburg_ast'] if (r['a'], r['b']) == tuple(NOVAS)][0]
    w('- **inverse_energy vs specaugment (v2, mesma maquina):** %s pp, IC 95%% [%s; %s], '
      'p = %s, positivo %d/%d\n' % (pp(rv2['est']), pp(rv2['lo']), pp(rv2['hi']), fp(rv2['p']),
                                     rv2['pos'], rv2['n']))
    m = o['msm']['homburg_ast']
    w('- **MSM vs controle, so as 4 originais (v1):** %s pp, IC 95%% [%s; %s], p = %s, '
      'positivo %d/%d sementes\n\n' % (pp(m['est']), pp(m['lo']), pp(m['hi']), fp(m['p']),
                                        m['pos'], m['n']))
    w('**Confundido com a maquina (nao interpretar como efeito de estrategia):** qualquer '
      'comparacao entre os grupos -- inverse_energy ou specaugment contra o controle ou contra '
      'qualquer estrategia original, e o Friedman das 6 estrategias. Valores so para registro:\n\n')
    w('| contraste (CONFUNDIDO com a maquina) | delta (pp) | IC 95% | p |\n|---|---|---|---|\n')
    for a in NOVAS:
        for b in HB_V1:
            r = resumo(delta_par(hb, a, b))
            w('| %s vs %s | %s | [%s; %s] | %s |\n' % (a, b, pp(r['est']), pp(r['lo']),
                                                      pp(r['hi']), fp(r['p'])))
    chi6, p6 = friedman(hb, ESTR)
    w('| Friedman 6 estrategias | chi2 = %.2f | | %s |\n\n' % (chi6, fp(p6)))

    # ---- Item 3: Holm
    w('## Todos os testes de hipotese, com Holm\n\n')
    w('Familias separadas, Holm aplicado uniformemente dentro de cada uma. Friedman: blocos = '
      'unidades do recorte. Pares: t pareado por unidade. Homburg: so pares da mesma maquina. '
      'AST k-fold: unidade = fold (1 semente), marcado com *.\n\n')
    fa = []
    for x in R:
        star = '*' if x['unidade'] == 'fold' else ''
        k, n, chi, p = o['friedman'][x['chave']]
        fa.append((x['rotulo'] + star, 'Friedman (%d estr., n=%d)' % (k, n), 'chi2=%.2f' % chi, p))
        for r in o['pares'][x['chave']]:
            fa.append((x['rotulo'] + star, '%s vs %s' % (r['a'], r['b']),
                       '%s pp' % pp(r['est']), r['p']))
    adj = holm([t[3] for t in fa])
    w('### (a) Estrategia vs estrategia -- %d testes\n\n' % len(fa))
    w('| recorte | teste | estatistica | p bruto | p Holm |\n|---|---|---|---|---|\n')
    for t, pa in sorted(zip(fa, adj), key=lambda z: z[0][3]):
        w('| %s | %s | %s | %s | %s |\n' % (t[0], t[1], t[2], fp(t[3]), fp(pa)))
    nb = sum(1 for t in fa if t[3] < 0.05)
    nh = sum(1 for pa in adj if pa < 0.05)
    w('\np bruto < 0,05: %d de %d (~%.1f esperados sob H0, ignorando a dependencia). '
      'p Holm < 0,05: **%d**.\n\n' % (nb, len(fa), 0.05 * len(fa), nh))

    fb = []
    for x in R:
        star = '*' if x['unidade'] == 'fold' else ''
        m = o['msm'][x['chave']]
        fb.append((x['rotulo'] + star, 'media de %d estr. - controle (n=%d)'
                   % (len(x['estr_msm']), m['n']), '%s pp' % pp(m['est']), m['p']))
    adjb = holm([t[3] for t in fb])
    w('### (b) MSM vs controle -- %d testes\n\n' % len(fb))
    w('| recorte | teste | delta | p bruto | p Holm |\n|---|---|---|---|---|\n')
    for t, pa in zip(fb, adjb):
        w('| %s | %s | %s | %s | %s |\n' % (t[0], t[1], t[2], fp(t[3]), fp(pa)))
    w('\nNota: com blocos semente x fold (n = 50) como no `analysis_ciclo2.md`, o CNN k-fold '
      'da Friedman p = 0,0745 e MSM p = 0,0028; aqui a unidade e a semente (n = 10), porque '
      'os folds de uma semente nao sao replicas independentes.\n\n')

    # ---- Item 4: meta-analise
    w('## Meta-analise de efeitos aleatorios\n\n')
    w('DerSimonian-Laird. IC principal: Hartung-Knapp-Sidik-Jonkman (t com k-1 gl, fator '
      'truncado em 1), recomendado para poucos estudos; o IC normal do DL aparece ao lado. '
      'Entrada: media e EP dos deltas por semente de cada recorte (EP de 6-10 sementes, '
      'tratado como conhecido -- mais uma razao para o IC de HKSJ).\n\n')
    w('**Recortes.** Principal: um recorte por base x arquitetura, sem reusar dados -- '
      'AST GTZAN fixo, CNN GTZAN k-fold, AST Homburg, AST FMA. CNN GTZAN fixo e k-fold usam '
      'as mesmas musicas e a mesma arquitetura; fica o k-fold (999 musicas de teste por '
      'semente, contra 150), e o fixo entra como sensibilidade. AST k-fold fica fora: uma '
      'semente so nao estima a variancia entre sementes, e ele reusa as musicas do AST fixo. '
      'Homburg: MSM vs controle so com as 4 originais; pares so dentro da mesma maquina.\n\n')
    nomes = {x['chave']: x['rotulo'] for x in R}
    w('### (a) MSM vs controle\n\n| analise | k | estimativa (pp) | IC 95% HKSJ | IC 95% DL | '
      'p (HKSJ) | tau2 (pp2) | I2 | Q (p) |\n|---|---|---|---|---|---|---|---|---|\n')
    for lab, key in (('principal', 'principal'), ('sensibilidade: CNN fixo no lugar do k-fold',
                                                  'sens_cnn_fixo'), ('so AST (3 bases)', 'so_ast')):
        r = o['meta_msm'][key]
        w('| %s | %d | %s | [%s; %s] | [%s; %s] | %s | %.2f | %.0f%% | %.2f (%s) |\n'
          % (lab, r['k'], pp(r['est']), pp(r['lo']), pp(r['hi']), pp(r['lo_dl']), pp(r['hi_dl']),
             fp(r['p']), 1e4 * r['tau2'], 100 * r['I2'], r['Q'], fp(r['pQ'])))
    w('\nEntradas (delta por semente, media e IC 95% t):\n\n| recorte | n | delta (pp) | IC 95% | '
      'p | positivo |\n|---|---|---|---|---|---|\n')
    for x in R:
        m = o['msm'][x['chave']]
        w('| %s%s | %d | %s | [%s; %s] | %s | %d/%d |\n'
          % (x['rotulo'], ' *' if x['unidade'] == 'fold' else '', m['n'], pp(m['est']), pp(m['lo']),
             pp(m['hi']), fp(m['p']), m['pos'], m['n']))

    w('\n### (b) Estrategia vs estrategia: a maior diferenca compativel com os dados\n\n')
    w('Medida: para cada par de estrategias, IC 95% **simultaneo** (Bonferroni sobre os m '
      'pares); X = o maior |limite| entre eles. Leitura: com 95% de confianca conjunta, nenhum '
      'par difere mais que X. Escolher o contraste de maior estimativa e dar a ele um IC comum '
      'seria viciado (selecao pelo maximo); o IC simultaneo corrige isso. No combinado, X usa '
      'os 6 pares das 4 estrategias originais, presentes nos 4 recortes (k = 4 por par). Os '
      'pares com inverse_energy/specaugment tem k = 2-3 (Homburg so tem o par entre elas), e '
      'HKSJ com 1-2 gl + Bonferroni da ICs sem sentido; eles ficam so na tabela por par.\n\n')
    w('| analise | pares | par que define X | estimativa (pp) | IC simultaneo | X (pp) | '
      'maior \\|estimativa\\| (IC 95% HKSJ) | tau2 (pp2) / I2 do par que define X |\n'
      '|---|---|---|---|---|---|---|---|\n')
    for lab, key in (('principal (4 originais)', 'principal'),
                     ('sensibilidade: CNN fixo', 'sens_cnn_fixo'), ('so AST (4 originais)', 'so_ast')):
        r = o['meta_pares'][key]
        t, g = r['top'], r['maior']
        w('| %s | %d | %s - %s | %s | [%s; %s] | %.2f | %s - %s: %s [%s; %s] | %.2f / %.0f%% |\n'
          % (lab, r['m'], t['par'][0], t['par'][1], pp(t['est']), pp(t['lo_sim']), pp(t['hi_sim']),
             100 * r['X'], g['par'][0], g['par'][1], pp(g['est']), pp(g['lo']), pp(g['hi']),
             1e4 * t['tau2'], 100 * t['I2']))
    w('\nTodos os pares combinados (recortes da analise principal; k varia):\n\n'
      '| par | k | estimativa (pp) | IC 95% HKSJ | IC 95% DL | tau2 (pp2) | I2 |\n'
      '|---|---|---|---|---|---|---|\n')
    for r in sorted(o['meta_pares']['todos_pares']['pares'], key=lambda r: -abs(r['est'])):
        w('| %s - %s | %d | %s | [%s; %s] | [%s; %s] | %.2f | %.0f%% |\n'
          % (r['par'][0], r['par'][1], r['k'], pp(r['est']), pp(r['lo']), pp(r['hi']),
             pp(r['lo_dl']), pp(r['hi_dl']), 1e4 * r['tau2'], 100 * r['I2']))
    w('\nSuplementar -- a pergunta original num contraste so: media das seletivas '
      '(spectral_energy, onset, combined) menos random, por semente:\n\n'
      '| recorte / analise | n ou k | delta (pp) | IC 95% | p | tau2 (pp2) / I2 |\n'
      '|---|---|---|---|---|---|\n')
    for x in R:
        r = o['sel'][x['chave']]
        w('| %s%s | %d | %s | [%s; %s] | %s | |\n'
          % (x['rotulo'], ' *' if x['unidade'] == 'fold' else '', r['n'], pp(r['est']),
             pp(r['lo']), pp(r['hi']), fp(r['p'])))
    for lab, key in (('combinado principal', 'principal'), ('combinado, CNN fixo', 'sens_cnn_fixo'),
                     ('combinado, so AST', 'so_ast')):
        r = o['meta_sel'][key]
        w('| **%s** | %d | %s | [%s; %s] | %s | %.2f / %.0f%% |\n'
          % (lab, r['k'], pp(r['est']), pp(r['lo']), pp(r['hi']), fp(r['p']),
             1e4 * r['tau2'], 100 * r['I2']))

    # ---- Item 5: frases
    w('\n## Limites de efeito ("excluido acima de X"), sem margem escolhida a posteriori\n\n')
    w('X = max(|limite inferior|, |limite superior|) do IC 95%; a decomposicao por direcao vem '
      'entre parenteses. Comparar com o DP entre sementes do recorte.\n\n')

    def frase_msm(m):
        X = 100 * max(abs(m['lo']), abs(m['hi']))
        if m['lo'] > 0 or m['hi'] < 0:
            dirs = 'o IC [%s; %s] exclui o zero' % (pp(m['lo']), pp(m['hi']))
        else:
            dirs = 'ganhos acima de %.2f pp e perdas acima de %.2f pp excluidos' % (
                100 * m['hi'], -100 * m['lo'])
        return X, dirs
    for x in R:
        m, d = o['msm'][x['chave']], o['dp'][x['chave']]
        mp, mo = o['maxpar'][x['chave']], o['maxpar_orig'][x['chave']]
        dps = 'n/a' if d != d else '%.2f pp' % (100 * d)
        X, dirs = frase_msm(m)
        w('- **%s**%s (DP entre sementes %s). MSM vs controle (%s pp): efeitos maiores que '
          '**%.2f pp** estao excluidos (%s). Entre estrategias: diferencas maiores que **%.2f pp** '
          'estao excluidas (%d pares; so as 4 originais: %.2f pp).\n'
          % (x['rotulo'], ' [unidade = fold, 1 semente]' if x['unidade'] == 'fold' else '', dps,
             pp(m['est']), X, dirs, 100 * mp['X'], mp['m'], 100 * mo['X']))
    for lab, key in (('Combinado (principal, k = 4)', 'principal'),
                     ('Combinado, so AST (k = 3)', 'so_ast')):
        r, rp = o['meta_msm'][key], o['meta_pares'][key]
        X, dirs = frase_msm(r)
        w('- **%s**. MSM vs controle (%s pp): efeitos maiores que **%.2f pp** estao excluidos '
          '(%s; HKSJ). Entre as 4 estrategias originais: diferencas maiores que **%.2f pp** '
          'estao excluidas (IC simultaneo, 6 pares).\n'
          % (lab, pp(r['est']), X, dirs, 100 * rp['X']))
    tip = np.nanmedian([o['dp'][x['chave']] for x in R])
    w('\nDP tipico entre sementes (mediana dos recortes): **%.2f pp**.\n' % (100 * tip))

    texto = buf.getvalue()
    with open(SAIDA, 'w', encoding='utf-8') as f:
        f.write(texto)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print(texto)


if __name__ == '__main__':
    main()
