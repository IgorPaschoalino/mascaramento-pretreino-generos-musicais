"""
analisar_remoto.py
==================
Tabela e testes estatisticos a partir de `resultados/resultados_remoto.json`.
Sem argumentos.

    python analisar_remoto.py

Funciona com resultados PARCIAIS: usa apenas as sementes em que o bloco esta
completo (controle + todas as estrategias), porque os testes empregados sao
testes por blocos. Sementes incompletas sao listadas e ignoradas, nao
silenciosamente misturadas.

Gera `resultados/analise_remoto.md` e imprime o mesmo conteudo.
"""

import os
import io
import sys
import json
import math
import itertools

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remoto_common import STRATEGIES, ESTRATEGIAS_ORIGINAIS

RESULTS_DIR = 'resultados'
RESULTS_FILE = os.path.join(RESULTS_DIR, 'resultados_remoto.json')
RUNS_DIR = os.path.join(RESULTS_DIR, 'execucoes')
SAIDA = os.path.join(RESULTS_DIR, 'analise_remoto.md')

TODAS_CONDICOES = ['controle'] + STRATEGIES


def condicoes_presentes(por_semente):
    """Condicoes que aparecem em ALGUMA semente do dataset, na ordem canonica.
    No Homburg da v2 so as 2 estrategias novas rodam aqui (o controle e as 4
    originais vem da corrida v1 e sao juntados na analise do TCC)."""
    vistas = set()
    for conds in por_semente.values():
        vistas.update(conds)
    return [c for c in TODAS_CONDICOES if c in vistas]


# ---------------------------------------------------------------------------
# Estatistica (mesmas funcoes usadas na analise do GTZAN)
# ---------------------------------------------------------------------------

def wilson(k, n, conf=0.95):
    if n == 0:
        return (float('nan'), float('nan'))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centro - meia, centro + meia)


def mde_two_proportions(p_base, n, alpha=0.05, power=0.80):
    """Menor diferenca detectavel entre duas proporcoes independentes."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    for delta in np.arange(0.0005, 1.0 - p_base, 0.0005):
        p2 = p_base + delta
        pbar = (p_base + p2) / 2
        se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
        se1 = math.sqrt((p_base * (1 - p_base) + p2 * (1 - p2)) / n)
        if delta >= z_a * se0 + z_b * se1:
            return float(delta)
    return float('nan')


def t_pareado(a, b):
    """t pareado que devolve nan em vez de um p espurio quando as diferencas tem
    variancia zero.

    Com poucas sementes e um conjunto de teste pequeno, e perfeitamente possivel
    que duas condicoes difiram pela MESMA quantidade em todas as sementes. O
    t de Student divide pelo desvio-padrao das diferencas e, sendo ele zero,
    devolve p = 0.0000 -- que parece significancia esmagadora e nao e nada: e
    uma divisao por zero. Nao pode chegar a uma tabela do TCC.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2:
        return float('nan')
    d = a - b
    if np.allclose(d.std(ddof=1), 0.0):
        return float('nan')
    return float(stats.ttest_rel(a, b).pvalue)


def mcnemar(y_true, pred_a, pred_b):
    """Teste pareado exato entre dois classificadores no MESMO conjunto de teste.

    Mais sensivel que comparar as duas acuracias como se fossem independentes,
    porque usa a concordancia exemplo a exemplo.
    """
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    n01 = int(np.sum(~a_ok & b_ok))
    n10 = int(np.sum(a_ok & ~b_ok))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)
    return n01, n10, p


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def carregar():
    if not os.path.exists(RESULTS_FILE):
        print('Nada para analisar: %s nao existe.' % RESULTS_FILE)
        print('Rode a corrida primeiro: python run_remoto.py')
        sys.exit(1)
    with open(RESULTS_FILE, encoding='utf-8') as f:
        bruto = json.load(f)

    # {dataset: {semente: {condicao: registro}}}
    org = {}
    for k, r in bruto.items():
        cond = 'controle' if r.get('condicao') == 'ctl' else r.get('estrategia')
        org.setdefault(r['dataset'], {}).setdefault(r['semente'], {})[cond] = r
    return org


def preds_de(dataset, semente, condicao):
    kind = 'ctl' if condicao == 'controle' else 'e2'
    estrategia = 'noadapt' if condicao == 'controle' else condicao
    caminho = os.path.join(RUNS_DIR, '%s-%s-%s-s%d'
                           % (dataset, kind, estrategia, semente), 'test_preds.json')
    if not os.path.exists(caminho):
        return None
    with open(caminho, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------

def relatar(org, out):
    w = out.write
    w('# Analise da execucao remota (FMA-small e Homburg)\n\n')
    w('Gerado por `analisar_remoto.py`. A regra metodologica do projeto e que\n')
    w('**nenhum resultado e considerado estabelecido antes de ser replicado com\n')
    w('o dobro das sementes** -- ela ja derrubou cinco resultados aparentes, e\n')
    w('acertou uma previsao. Leia as secoes abaixo com isso em mente.\n\n')

    amb = os.path.join(RESULTS_DIR, 'ambiente.json')
    if os.path.exists(amb):
        with open(amb, encoding='utf-8') as f:
            a = json.load(f)
        w('Ambiente: %s | torch %s | %s\n\n'
          % (a.get('gpu', '?'), a.get('torch', '?'), a.get('plataforma', '?')))

    for dataset in sorted(org):
        CONDICOES = condicoes_presentes(org[dataset])
        ESTR = [c for c in CONDICOES if c != 'controle']
        tem_ctl = 'controle' in CONDICOES
        sementes_all = sorted(org[dataset])
        completas = [s for s in sementes_all
                     if all(c in org[dataset][s] for c in CONDICOES)]
        incompletas = [s for s in sementes_all if s not in completas]

        w('## %s\n\n' % dataset.upper())
        if incompletas:
            w('Sementes incompletas, IGNORADAS nos testes (faltam condicoes): %s\n\n'
              % ', '.join(str(s) for s in incompletas))
        if not completas:
            w('Nenhuma semente completa ainda -- nada a testar.\n\n')
            continue
        w('Sementes completas: **%d** (%s)\n\n'
          % (len(completas), ', '.join(str(s) for s in completas)))

        n_test = org[dataset][completas[0]][CONDICOES[0]].get('n_test', 0)
        if not tem_ctl:
            w('Sem controle nesta corrida (reaproveitado da corrida anterior): '
              'so a comparacao entre as estrategias presentes.\n\n')

        # ---- tabela por condicao ----
        w('### Acuracia por condicao\n\n')
        # Sem formatacao % nesta linha, entao o sinal de porcentagem e simples.
        w('| condicao | media | desvio-padrao | IC 95% da media | min | max |\n')
        w('|---|---|---|---|---|---|\n')
        medias = {}
        for cond in CONDICOES:
            vals = np.array([org[dataset][s][cond]['test_accuracy'] for s in completas])
            medias[cond] = vals
            if len(vals) > 1:
                sem = stats.sem(vals)
                ic = stats.t.interval(0.95, len(vals) - 1, loc=vals.mean(), scale=sem) \
                    if sem > 0 else (vals.mean(), vals.mean())
            else:
                ic = (float('nan'), float('nan'))
            w('| %s | %.4f | %.4f | [%.3f; %.3f] | %.4f | %.4f |\n'
              % (cond, vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0,
                 ic[0], ic[1], vals.min(), vals.max()))
        w('\n')

        # ---- poder ----
        p_base = float(medias[CONDICOES[0]].mean())
        mde = mde_two_proportions(p_base, n_test) if n_test else float('nan')
        w('### Poder estatistico\n\n')
        w('- Conjunto de teste: **%d** exemplos (um acerto = %.2f pp)\n'
          % (n_test, 100.0 / n_test if n_test else float('nan')))
        if math.isnan(mde):
            w('- Menor diferenca detectavel: **indeterminada** para n = %d '
              '(conjunto de teste pequeno demais para a aproximacao normal)\n' % n_test)
        else:
            w('- Menor diferenca detectavel (alpha 0,05, poder 0,80): **%.1f pp**\n'
              % (100 * mde))
        w('- Referencia do GTZAN: 150 exemplos, MDE 10,4 pp\n')
        if n_test and not math.isnan(mde) and mde < 0.104:
            w('- Este dataset resolve diferencas **menores** que o GTZAN.\n')
        else:
            w('- Este dataset **nao** resolve diferencas menores que o GTZAN '
              '(acuracia proxima de 0,5 e onde a variancia e maxima).\n')
        w('\n')

        # ---- Friedman entre as 4 estrategias ----
        w('### As estrategias se separam? (Friedman, blocos = sementes)\n\n')
        if len(completas) >= 3 and len(ESTR) >= 3:
            grupos = [('todas as %d estrategias' % len(ESTR), ESTR)]
            if (len(ESTR) > len(ESTRATEGIAS_ORIGINAIS)
                    and all(e in ESTR for e in ESTRATEGIAS_ORIGINAIS)):
                grupos.append(('so as 4 originais', ESTRATEGIAS_ORIGINAIS))
            for rotulo, grupo in grupos:
                chi2, p = stats.friedmanchisquare(*[medias[s] for s in grupo])
                w('- %s: chi2 = %.3f, **p = %.4f** (%d sementes) -- %s\n'
                  % (rotulo, chi2, p, len(completas),
                     'ha diferenca detectavel' if p < 0.05
                     else 'NAO ha diferenca detectavel'))
        elif len(ESTR) < 3:
            w('- Menos de 3 estrategias nesta corrida -- ver os pares abaixo.\n')
        else:
            w('- Friedman exige pelo menos 3 sementes completas (ha %d).\n' % len(completas))
        w('\n')

        # ---- pares de estrategias ----
        if len(completas) >= 2:
            w('#### Pares de estrategias (t pareado por semente)\n\n')
            w('| par | delta medio (pp) | p |\n|---|---|---|\n')
            linhas = []
            for a, b in itertools.combinations(ESTR, 2):
                d = medias[a] - medias[b]
                p = t_pareado(medias[a], medias[b])
                linhas.append((p, a, b, 100 * d.mean()))
            for p, a, b, delta in sorted(linhas, key=lambda x: (np.isnan(x[0]), x[0])):
                w('| %s vs %s | %+.2f | %s |\n'
                  % (a, b, delta, 'degenerado' if np.isnan(p) else '%.4f' % p))
            w('\n')

        if tem_ctl:
            # ---- MSM vs controle ----
            w('### O MSM ajuda? (cada estrategia contra o controle)\n\n')
            w('| estrategia | delta medio (pp) | positivo em | p (t pareado) |\n')
            w('|---|---|---|---|\n')
            todos_deltas = []
            for s in ESTR:
                d = medias[s] - medias['controle']
                todos_deltas.append(d)
                p = t_pareado(medias[s], medias['controle'])
                w('| %s | %+.2f | %d de %d | %s |\n'
                  % (s, 100 * d.mean(), int((d > 0).sum()), len(d),
                     'degenerado' if np.isnan(p) else '%.4f' % p))
            w('\n')

            agregado = np.concatenate(todos_deltas)
            w('Agregado das %d comparacoes (estrategias x %d sementes): '
              '**%+.2f pp**, positivo em %d de %d.\n\n'
              % (len(agregado), len(completas), 100 * agregado.mean(),
                 int((agregado > 0).sum()), len(agregado)))
            if len(completas) > 1:
                # Teste conservador: media por semente, nao as 4x N comparacoes como
                # se fossem independentes -- as 4 estrategias de uma mesma semente
                # compartilham o mesmo controle, entao nao sao independentes.
                por_semente = np.mean(np.vstack(todos_deltas), axis=0)
                p_cons = t_pareado(por_semente, np.zeros_like(por_semente))
                w('Teste **conservador** (media por semente, n = %d, porque as 4 '
                  'estrategias de uma semente dividem o mesmo controle e nao sao '
                  'independentes): **p = %s**.\n\n'
                  % (len(por_semente),
                     'degenerado' if np.isnan(p_cons) else '%.4f' % p_cons))

            # ---- McNemar ----
            pares_mc = []
            for s in completas:
                pc = preds_de(dataset, s, 'controle')
                if pc is None:
                    continue
                for e in ESTR:
                    pe = preds_de(dataset, s, e)
                    if pe is None:
                        continue
                    n01, n10, p = mcnemar(pc['y_true'], pc['y_pred'], pe['y_pred'])
                    pares_mc.append((dataset, s, e, n01, n10, p))
            if pares_mc:
                w('#### McNemar exato, estrategia vs controle, exemplo a exemplo\n\n')
                w('Teste pareado de verdade (usa a concordancia por exemplo, nao so '
                  'as duas acuracias). `so_estrategia` = exemplos que so a '
                  'estrategia acertou.\n\n')
                w('| semente | estrategia | so_estrategia | so_controle | p |\n')
                w('|---|---|---|---|---|\n')
                for _, s, e, n01, n10, p in pares_mc:
                    w('| %d | %s | %d | %d | %.4f |\n' % (s, e, n01, n10, p))
                sig = [x for x in pares_mc if x[5] < 0.05]
                w('\n%d de %d comparacoes significativas a 5%%%s.\n\n'
                  % (len(sig), len(pares_mc),
                     '' if not sig else ' -- mas com %d testes, ~%.1f falsos '
                     'positivos sao esperados por acaso' % (len(pares_mc), 0.05 * len(pares_mc))))

        # ---- o pretexto aprendeu? ----
        quedas = []
        for s in completas:
            for e in ESTR:
                r = org[dataset][s][e]
                if 'msm_val_loss_first' in r and 'msm_val_loss_last' in r:
                    f0, f1 = r['msm_val_loss_first'], r['msm_val_loss_last']
                    if f0:
                        quedas.append(100 * (1 - f1 / f0))
        if quedas:
            w('### O pretexto MSM aprendeu alguma coisa?\n\n')
            w('Queda da perda de reconstrucao na validacao, do inicio ao fim do '
              'Estagio 1: media **%.1f%%** (min %.1f%%, max %.1f%%, n = %d).\n\n'
              % (np.mean(quedas), np.min(quedas), np.max(quedas), len(quedas)))
            w('Isso importa para a interpretacao: se a perda NAO cai, um '
              'resultado nulo pode significar apenas que o pre-treino nao '
              'rodou de fato. Se cai, o nulo e sobre o metodo.\n\n')

        # ---- veredito ----
        w('### Leitura\n\n')
        if len(completas) < 6:
            w('**%d sementes completas.** Sob a regra do projeto, este resultado '
              'ainda NAO esta estabelecido -- e preciso chegar a 6 sementes, e '
              'depois confirmar em 12 se algo aparecer. Tres resultados deste '
              'projeto se inverteram exatamente nesta faixa.\n\n' % len(completas))
        else:
            w('**%d sementes completas**, mesmo numero do AST no GTZAN. '
              'Comparavel diretamente com o resultado principal do trabalho.\n\n'
              % len(completas))

        # ---- custo ----
        mins = [org[dataset][s][c].get('minutos', 0) for s in completas for c in CONDICOES]
        if mins:
            w('Custo: %.1f h de GPU em %d execucoes (mediana %.0f min cada).\n\n'
              % (sum(mins) / 60.0, len(mins), float(np.median(mins))))

    # ---- comparacao com o GTZAN ----
    w('## Referencia: o que o GTZAN ja dizia\n\n')
    w('| | AST no GTZAN (6 sementes) | CNN no GTZAN (10 sementes) |\n')
    w('|---|---|---|\n')
    w('| MSM vs controle | +0,81 pp, p conservador = 0,481 | -0,70 pp, p = 0,111 |\n')
    w('| Friedman entre estrategias | p = 0,309 | p = 0,315 |\n')
    w('| Conjunto de teste | 150 exemplos, MDE 10,4 pp | 150 exemplos |\n\n')
    w('As duas arquiteturas concordam: no GTZAN o MSM nao produz efeito '
      'detectavel e as estrategias nao se separam. A pergunta que esta corrida '
      'responde e se isso **generaliza** para outros datasets e taxonomias.\n')


def main():
    org = carregar()
    buf = io.StringIO()
    relatar(org, buf)
    texto = buf.getvalue()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SAIDA, 'w', encoding='utf-8') as f:
        f.write(texto)
    # UTF-8 explicito: o console do Windows quebra com acentos em cp1252.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print(texto)
    print('-' * 70)
    print('Gravado em %s' % SAIDA)


if __name__ == '__main__':
    main()
