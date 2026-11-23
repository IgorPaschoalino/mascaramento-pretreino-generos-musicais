# Analise final -- Homburg por maquina, Holm, meta-analise, limites de efeito

Gerado por `analysis_final.py` (sem argumentos; le os JSONs de resultados). Unidade: semente (k-fold: semente com os 5 folds agregados, 999 musicas). Deltas em pontos percentuais (pp). Nada aqui envolveu treino novo.

## Recortes

| recorte | unidade | n | condicoes | DP entre sementes (pp) |
|---|---|---|---|---|
| GTZAN · AST · fixo | semente | 6 | 7 | 2.19 |
| GTZAN · CNN · fixo | semente | 10 | 7 | 1.54 |
| GTZAN · CNN · k-fold | semente | 10 | 7 | 0.69 |
| GTZAN · AST · k-fold (1 sem.) | fold | 5 | 7 | n/a (1 semente) |
| Homburg · AST · fixo | semente | 6 | 7 | 1.83 |
| FMA-small · AST · fixo | semente | 6 | 7 | 1.36 |

DP entre sementes = raiz da media, entre condicoes, da variancia da acuracia entre sementes. E o "ruido de trocar a semente": um efeito menor que ele nao tem valor pratico.

## Homburg separado por maquina

v1 (RTX 5060): controle + random, spectral_energy, onset, combined. v2 (RTX 5090): inverse_energy, specaugment. Mesmo conjunto de teste (285).

- **Friedman, 4 originais (v1, mesma maquina):** chi2 = 1.60, p = 0.6586 (n = 6 sementes)
- **inverse_energy vs specaugment (v2, mesma maquina):** +1.23 pp, IC 95% [-1.11; +3.57], p = 0.2348, positivo 5/6
- **MSM vs controle, so as 4 originais (v1):** -0.47 pp, IC 95% [-3.73; +2.79], p = 0.7274, positivo 2/6 sementes

**Confundido com a maquina (nao interpretar como efeito de estrategia):** qualquer comparacao entre os grupos -- inverse_energy ou specaugment contra o controle ou contra qualquer estrategia original, e o Friedman das 6 estrategias. Valores so para registro:

| contraste (CONFUNDIDO com a maquina) | delta (pp) | IC 95% | p |
|---|---|---|---|
| inverse_energy vs controle | +0.70 | [-2.02; +3.43] | 0.5374 |
| inverse_energy vs random | +1.05 | [-0.64; +2.75] | 0.1714 |
| inverse_energy vs spectral_energy | +1.40 | [+0.36; +2.45] | 0.0180 |
| inverse_energy vs onset | +1.40 | [-0.47; +3.28] | 0.1127 |
| inverse_energy vs combined | +0.82 | [-1.00; +2.63] | 0.2983 |
| specaugment vs controle | -0.53 | [-5.46; +4.41] | 0.7949 |
| specaugment vs random | -0.18 | [-2.98; +2.63] | 0.8784 |
| specaugment vs spectral_energy | +0.18 | [-2.24; +2.59] | 0.8593 |
| specaugment vs onset | +0.18 | [-1.31; +1.66] | 0.7738 |
| specaugment vs combined | -0.41 | [-2.61; +1.79] | 0.6529 |
| Friedman 6 estrategias | chi2 = 4.20 | | 0.5211 |

## Todos os testes de hipotese, com Holm

Familias separadas, Holm aplicado uniformemente dentro de cada uma. Friedman: blocos = unidades do recorte. Pares: t pareado por unidade. Homburg: so pares da mesma maquina. AST k-fold: unidade = fold (1 semente), marcado com *.

### (a) Estrategia vs estrategia -- 88 testes

| recorte | teste | estatistica | p bruto | p Holm |
|---|---|---|---|---|
| GTZAN · CNN · k-fold | random vs specaugment | +0.53 pp | 0.0005 | 0.0436 |
| FMA-small · AST · fixo | onset vs specaugment | +2.27 pp | 0.0066 | 0.5708 |
| FMA-small · AST · fixo | spectral_energy vs onset | -2.31 pp | 0.0110 | 0.9484 |
| GTZAN · CNN · k-fold | inverse_energy vs specaugment | +0.84 pp | 0.0127 | 1.0000 |
| GTZAN · CNN · k-fold | combined vs specaugment | +0.83 pp | 0.0133 | 1.0000 |
| FMA-small · AST · fixo | Friedman (6 estr., n=6) | chi2=12.22 | 0.0318 | 1.0000 |
| FMA-small · AST · fixo | random vs onset | -1.42 pp | 0.0647 | 1.0000 |
| FMA-small · AST · fixo | onset vs combined | +1.52 pp | 0.0683 | 1.0000 |
| FMA-small · AST · fixo | inverse_energy vs specaugment | +1.00 pp | 0.0772 | 1.0000 |
| GTZAN · CNN · k-fold | Friedman (6 estr., n=10) | chi2=9.67 | 0.0852 | 1.0000 |
| GTZAN · CNN · k-fold | spectral_energy vs specaugment | +0.61 pp | 0.0856 | 1.0000 |
| GTZAN · CNN · fixo | spectral_energy vs inverse_energy | -1.07 pp | 0.0954 | 1.0000 |
| FMA-small · AST · fixo | spectral_energy vs inverse_energy | -1.04 pp | 0.1006 | 1.0000 |
| GTZAN · CNN · fixo | inverse_energy vs specaugment | +0.67 pp | 0.1066 | 1.0000 |
| FMA-small · AST · fixo | onset vs inverse_energy | +1.27 pp | 0.1162 | 1.0000 |
| GTZAN · AST · fixo | spectral_energy vs onset | -1.44 pp | 0.1434 | 1.0000 |
| GTZAN · CNN · fixo | spectral_energy vs onset | -1.27 pp | 0.1605 | 1.0000 |
| GTZAN · AST · fixo | onset vs combined | +2.00 pp | 0.1613 | 1.0000 |
| GTZAN · AST · fixo | onset vs inverse_energy | +1.67 pp | 0.1614 | 1.0000 |
| GTZAN · CNN · fixo | random vs spectral_energy | +0.80 pp | 0.1744 | 1.0000 |
| GTZAN · CNN · k-fold | onset vs specaugment | +0.50 pp | 0.1828 | 1.0000 |
| GTZAN · CNN · k-fold | onset vs inverse_energy | -0.34 pp | 0.1832 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | random vs onset | -2.60 pp | 0.2002 | 1.0000 |
| FMA-small · AST · fixo | spectral_energy vs combined | -0.79 pp | 0.2037 | 1.0000 |
| FMA-small · AST · fixo | random vs specaugment | +0.85 pp | 0.2154 | 1.0000 |
| GTZAN · AST · fixo | random vs onset | -0.89 pp | 0.2213 | 1.0000 |
| FMA-small · AST · fixo | random vs spectral_energy | +0.90 pp | 0.2309 | 1.0000 |
| GTZAN · CNN · fixo | Friedman (6 estr., n=10) | chi2=6.85 | 0.2324 | 1.0000 |
| Homburg · AST · fixo | inverse_energy vs specaugment | +1.23 pp | 0.2348 | 1.0000 |
| GTZAN · CNN · fixo | onset vs combined | +0.80 pp | 0.2393 | 1.0000 |
| GTZAN · CNN · fixo | onset vs specaugment | +0.87 pp | 0.2445 | 1.0000 |
| GTZAN · AST · fixo | combined vs specaugment | -0.89 pp | 0.2739 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | random vs specaugment | -2.40 pp | 0.2855 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | inverse_energy vs specaugment | -1.60 pp | 0.2917 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | random vs spectral_energy | -1.50 pp | 0.2944 | 1.0000 |
| FMA-small · AST · fixo | combined vs specaugment | +0.75 pp | 0.3118 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | spectral_energy vs onset | -1.10 pp | 0.3126 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | random vs combined | -1.70 pp | 0.3393 | 1.0000 |
| GTZAN · CNN · k-fold | random vs inverse_energy | -0.31 pp | 0.3538 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | onset vs inverse_energy | +1.80 pp | 0.3560 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | onset vs combined | +0.90 pp | 0.3739 | 1.0000 |
| GTZAN · AST · fixo | onset vs specaugment | +1.11 pp | 0.3881 | 1.0000 |
| GTZAN · CNN · fixo | random vs onset | -0.47 pp | 0.3901 | 1.0000 |
| Homburg · AST · fixo | onset vs combined | -0.58 pp | 0.3927 | 1.0000 |
| GTZAN · CNN · k-fold | random vs combined | -0.30 pp | 0.3974 | 1.0000 |
| GTZAN · CNN · fixo | combined vs inverse_energy | -0.60 pp | 0.4174 | 1.0000 |
| GTZAN · AST · fixo | Friedman (6 estr., n=6) | chi2=4.95 | 0.4221 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | spectral_energy vs inverse_energy | +0.70 pp | 0.4383 | 1.0000 |
| GTZAN · AST · fixo | random vs inverse_energy | +0.78 pp | 0.4499 | 1.0000 |
| GTZAN · CNN · k-fold | onset vs combined | -0.33 pp | 0.4542 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | spectral_energy vs specaugment | -0.90 pp | 0.4573 | 1.0000 |
| GTZAN · CNN · k-fold | spectral_energy vs inverse_energy | -0.23 pp | 0.4788 | 1.0000 |
| GTZAN · AST · fixo | random vs combined | +1.11 pp | 0.5166 | 1.0000 |
| GTZAN · CNN · fixo | spectral_energy vs combined | -0.47 pp | 0.5352 | 1.0000 |
| GTZAN · CNN · fixo | random vs specaugment | +0.40 pp | 0.5363 | 1.0000 |
| GTZAN · CNN · fixo | spectral_energy vs specaugment | -0.40 pp | 0.5509 | 1.0000 |
| GTZAN · CNN · k-fold | spectral_energy vs combined | -0.22 pp | 0.5528 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | random vs inverse_energy | -0.80 pp | 0.5694 | 1.0000 |
| Homburg · AST · fixo | spectral_energy vs combined | -0.58 pp | 0.5713 | 1.0000 |
| FMA-small · AST · fixo | combined vs inverse_energy | -0.25 pp | 0.6003 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | combined vs inverse_energy | +0.90 pp | 0.6044 | 1.0000 |
| GTZAN · AST · fixo | random vs spectral_energy | +0.56 pp | 0.6157 | 1.0000 |
| Homburg · AST · fixo | random vs spectral_energy | +0.35 pp | 0.6436 | 1.0000 |
| GTZAN · AST · fixo | spectral_energy vs combined | +0.56 pp | 0.6450 | 1.0000 |
| Homburg · AST · fixo | Friedman (4 estr., n=6) | chi2=1.60 | 0.6586 | 1.0000 |
| GTZAN · CNN · fixo | random vs combined | +0.33 pp | 0.6733 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | combined vs specaugment | -0.70 pp | 0.6737 | 1.0000 |
| GTZAN · CNN · fixo | random vs inverse_energy | -0.27 pp | 0.6744 | 1.0000 |
| GTZAN · AST · fixo | inverse_energy vs specaugment | -0.56 pp | 0.7144 | 1.0000 |
| Homburg · AST · fixo | random vs onset | +0.35 pp | 0.7234 | 1.0000 |
| GTZAN · CNN · k-fold | spectral_energy vs onset | +0.11 pp | 0.7335 | 1.0000 |
| GTZAN · AST · fixo | spectral_energy vs specaugment | -0.33 pp | 0.7591 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | Friedman (6 estr., n=5) | chi2=2.49 | 0.7787 | 1.0000 |
| GTZAN · CNN · fixo | onset vs inverse_energy | +0.20 pp | 0.7941 | 1.0000 |
| Homburg · AST · fixo | random vs combined | -0.23 pp | 0.8033 | 1.0000 |
| GTZAN · CNN · k-fold | random vs spectral_energy | -0.08 pp | 0.8098 | 1.0000 |
| GTZAN · AST · fixo | spectral_energy vs inverse_energy | +0.22 pp | 0.8220 | 1.0000 |
| GTZAN · AST · fixo | combined vs inverse_energy | -0.33 pp | 0.8246 | 1.0000 |
| FMA-small · AST · fixo | random vs inverse_energy | -0.15 pp | 0.8587 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | spectral_energy vs combined | -0.20 pp | 0.8610 | 1.0000 |
| GTZAN · AST · k-fold (1 sem.)* | onset vs specaugment | +0.20 pp | 0.8794 | 1.0000 |
| GTZAN · AST · fixo | random vs specaugment | +0.22 pp | 0.9012 | 1.0000 |
| FMA-small · AST · fixo | random vs combined | +0.10 pp | 0.9085 | 1.0000 |
| GTZAN · CNN · fixo | combined vs specaugment | +0.07 pp | 0.9088 | 1.0000 |
| GTZAN · CNN · k-fold | random vs onset | +0.03 pp | 0.9382 | 1.0000 |
| FMA-small · AST · fixo | spectral_energy vs specaugment | -0.04 pp | 0.9486 | 1.0000 |
| GTZAN · CNN · k-fold | combined vs inverse_energy | -0.01 pp | 0.9731 | 1.0000 |
| Homburg · AST · fixo | spectral_energy vs onset | +0.00 pp | 1.0000 | 1.0000 |

p bruto < 0,05: 6 de 88 (~4.4 esperados sob H0, ignorando a dependencia). p Holm < 0,05: **1**.

### (b) MSM vs controle -- 6 testes

| recorte | teste | delta | p bruto | p Holm |
|---|---|---|---|---|
| GTZAN · AST · fixo | media de 6 estr. - controle (n=6) | +0.70 pp | 0.5226 | 1.0000 |
| GTZAN · CNN · fixo | media de 6 estr. - controle (n=10) | -0.67 pp | 0.1954 | 0.7816 |
| GTZAN · CNN · k-fold | media de 6 estr. - controle (n=10) | +0.80 pp | 0.0132 | 0.0793 |
| GTZAN · AST · k-fold (1 sem.)* | media de 6 estr. - controle (n=5) | +1.30 pp | 0.3901 | 1.0000 |
| Homburg · AST · fixo | media de 4 estr. - controle (n=6) | -0.47 pp | 0.7274 | 1.0000 |
| FMA-small · AST · fixo | media de 6 estr. - controle (n=6) | -1.97 pp | 0.1104 | 0.5521 |

Nota: com blocos semente x fold (n = 50) como no `analysis_ciclo2.md`, o CNN k-fold da Friedman p = 0,0745 e MSM p = 0,0028; aqui a unidade e a semente (n = 10), porque os folds de uma semente nao sao replicas independentes.

## Meta-analise de efeitos aleatorios

DerSimonian-Laird. IC principal: Hartung-Knapp-Sidik-Jonkman (t com k-1 gl, fator truncado em 1), recomendado para poucos estudos; o IC normal do DL aparece ao lado. Entrada: media e EP dos deltas por semente de cada recorte (EP de 6-10 sementes, tratado como conhecido -- mais uma razao para o IC de HKSJ).

**Recortes.** Principal: um recorte por base x arquitetura, sem reusar dados -- AST GTZAN fixo, CNN GTZAN k-fold, AST Homburg, AST FMA. CNN GTZAN fixo e k-fold usam as mesmas musicas e a mesma arquitetura; fica o k-fold (999 musicas de teste por semente, contra 150), e o fixo entra como sensibilidade. AST k-fold fica fora: uma semente so nao estima a variancia entre sementes, e ele reusa as musicas do AST fixo. Homburg: MSM vs controle so com as 4 originais; pares so dentro da mesma maquina.

### (a) MSM vs controle

| analise | k | estimativa (pp) | IC 95% HKSJ | IC 95% DL | p (HKSJ) | tau2 (pp2) | I2 | Q (p) |
|---|---|---|---|---|---|---|---|---|
| principal | 4 | -0.03 | [-2.15; +2.09] | [-1.33; +1.28] | 0.9697 | 1.03 | 61% | 7.72 (0.0522) |
| sensibilidade: CNN fixo no lugar do k-fold | 4 | -0.64 | [-2.05; +0.78] | [-1.49; +0.22] | 0.2481 | 0.12 | 13% | 3.45 (0.3274) |
| so AST (3 bases) | 3 | -0.59 | [-4.16; +2.98] | [-2.22; +1.03] | 0.5502 | 0.87 | 42% | 3.44 (0.1788) |

Entradas (delta por semente, media e IC 95% t):

| recorte | n | delta (pp) | IC 95% | p | positivo |
|---|---|---|---|---|---|
| GTZAN · AST · fixo | 6 | +0.70 | [-1.93; +3.34] | 0.5226 | 4/6 |
| GTZAN · CNN · fixo | 10 | -0.67 | [-1.74; +0.41] | 0.1954 | 4/10 |
| GTZAN · CNN · k-fold | 10 | +0.80 | [+0.21; +1.39] | 0.0132 | 8/10 |
| GTZAN · AST · k-fold (1 sem.) * | 5 | +1.30 | [-2.45; +5.06] | 0.3901 | 3/5 |
| Homburg · AST · fixo | 6 | -0.47 | [-3.73; +2.79] | 0.7274 | 2/6 |
| FMA-small · AST · fixo | 6 | -1.97 | [-4.57; +0.64] | 0.1104 | 1/6 |

### (b) Estrategia vs estrategia: a maior diferenca compativel com os dados

Medida: para cada par de estrategias, IC 95% **simultaneo** (Bonferroni sobre os m pares); X = o maior |limite| entre eles. Leitura: com 95% de confianca conjunta, nenhum par difere mais que X. Escolher o contraste de maior estimativa e dar a ele um IC comum seria viciado (selecao pelo maximo); o IC simultaneo corrige isso. No combinado, X usa os 6 pares das 4 estrategias originais, presentes nos 4 recortes (k = 4 por par). Os pares com inverse_energy/specaugment tem k = 2-3 (Homburg so tem o par entre elas), e HKSJ com 1-2 gl + Bonferroni da ICs sem sentido; eles ficam so na tabela por par.

| analise | pares | par que define X | estimativa (pp) | IC simultaneo | X (pp) | maior \|estimativa\| (IC 95% HKSJ) | tau2 (pp2) / I2 do par que define X |
|---|---|---|---|---|---|---|---|
| principal (4 originais) | 6 | onset - spectral_energy | +0.88 | [-3.26; +5.02] | 5.02 | onset - spectral_energy: +0.88 [-1.24; +2.99] | 1.35 / 80% |
| sensibilidade: CNN fixo | 6 | onset - spectral_energy | +1.34 | [-1.83; +4.52] | 4.52 | onset - spectral_energy: +1.34 [-0.28; +2.96] | 0.47 / 45% |
| so AST (4 originais) | 6 | combined - onset | -0.83 | [-10.02; +8.36] | 10.02 | onset - spectral_energy: +1.33 [-1.68; +4.34] | 1.47 / 71% |

Todos os pares combinados (recortes da analise principal; k varia):

| par | k | estimativa (pp) | IC 95% HKSJ | IC 95% DL | tau2 (pp2) | I2 |
|---|---|---|---|---|---|---|
| onset - specaugment | 3 | +1.29 | [-1.66; +4.24] | [-0.05; +2.63] | 0.99 | 76% |
| onset - spectral_energy | 4 | +0.88 | [-1.24; +2.99] | [-0.42; +2.18] | 1.35 | 80% |
| inverse_energy - specaugment | 4 | +0.87 | [+0.16; +1.58] | [+0.43; +1.31] | 0.00 | 0% |
| inverse_energy - onset | 3 | -0.66 | [-3.73; +2.40] | [-2.06; +0.73] | 1.11 | 76% |
| random - specaugment | 3 | +0.54 | [+0.11; +0.96] | [+0.34; +0.73] | 0.00 | 0% |
| onset - random | 4 | +0.50 | [-0.77; +1.77] | [-0.28; +1.28] | 0.28 | 44% |
| combined - onset | 4 | -0.42 | [-2.38; +1.53] | [-1.57; +0.72] | 0.88 | 68% |
| specaugment - spectral_energy | 3 | -0.42 | [-1.59; +0.75] | [-0.95; +0.11] | 0.00 | 0% |
| inverse_energy - spectral_energy | 3 | +0.41 | [-0.84; +1.67] | [-0.16; +0.98] | 0.04 | 12% |
| combined - specaugment | 3 | +0.38 | [-1.86; +2.61] | [-0.59; +1.34] | 0.44 | 60% |
| combined - spectral_energy | 4 | +0.35 | [-0.53; +1.23] | [-0.19; +0.89] | 0.00 | 0% |
| combined - random | 4 | +0.20 | [-0.73; +1.13] | [-0.37; +0.77] | 0.00 | 0% |
| inverse_energy - random | 3 | +0.19 | [-1.01; +1.40] | [-0.36; +0.74] | 0.00 | 0% |
| random - spectral_energy | 4 | +0.17 | [-0.66; +1.00] | [-0.34; +0.68] | 0.00 | 0% |
| combined - inverse_energy | 3 | -0.09 | [-1.12; +0.94] | [-0.56; +0.38] | 0.00 | 0% |

Suplementar -- a pergunta original num contraste so: media das seletivas (spectral_energy, onset, combined) menos random, por semente:

| recorte / analise | n ou k | delta (pp) | IC 95% | p | tau2 (pp2) / I2 |
|---|---|---|---|---|---|
| GTZAN · AST · fixo | 6 | -0.26 | [-2.77; +2.25] | 0.8015 | |
| GTZAN · CNN · fixo | 10 | -0.22 | [-1.24; +0.80] | 0.6337 | |
| GTZAN · CNN · k-fold | 10 | +0.12 | [-0.50; +0.74] | 0.6803 | |
| GTZAN · AST · k-fold (1 sem.) * | 5 | +1.94 | [-1.97; +5.84] | 0.2410 | |
| Homburg · AST · fixo | 6 | -0.16 | [-1.99; +1.68] | 0.8355 | |
| FMA-small · AST · fixo | 6 | +0.14 | [-1.47; +1.75] | 0.8333 | |
| **combinado principal** | 4 | +0.07 | [-0.66; +0.80] | 0.7797 | 0.00 / 0% |
| **combinado, CNN fixo** | 4 | -0.13 | [-1.11; +0.86] | 0.7112 | 0.00 / 0% |
| **combinado, so AST** | 3 | -0.04 | [-1.86; +1.78] | 0.9330 | 0.00 / 0% |

## Limites de efeito ("excluido acima de X"), sem margem escolhida a posteriori

X = max(|limite inferior|, |limite superior|) do IC 95%; a decomposicao por direcao vem entre parenteses. Comparar com o DP entre sementes do recorte.

- **GTZAN · AST · fixo** (DP entre sementes 2.19 pp). MSM vs controle (+0.70 pp): efeitos maiores que **3.34 pp** estao excluidos (ganhos acima de 3.34 pp e perdas acima de 1.93 pp excluidos). Entre estrategias: diferencas maiores que **9.47 pp** estao excluidas (15 pares; so as 4 originais: 7.83 pp).
- **GTZAN · CNN · fixo** (DP entre sementes 1.54 pp). MSM vs controle (-0.67 pp): efeitos maiores que **1.74 pp** estao excluidos (ganhos acima de 0.41 pp e perdas acima de 1.74 pp excluidos). Entre estrategias: diferencas maiores que **4.54 pp** estao excluidas (15 pares; so as 4 originais: 4.05 pp).
- **GTZAN · CNN · k-fold** (DP entre sementes 0.69 pp). MSM vs controle (+0.80 pp): efeitos maiores que **1.39 pp** estao excluidos (o IC [+0.21; +1.39] exclui o zero). Entre estrategias: diferencas maiores que **2.00 pp** estao excluidas (15 pares; so as 4 originais: 1.75 pp).
- **GTZAN · AST · k-fold (1 sem.)** [unidade = fold, 1 semente] (DP entre sementes n/a). MSM vs controle (+1.30 pp): efeitos maiores que **5.06 pp** estao excluidos (ganhos acima de 5.06 pp e perdas acima de 2.45 pp excluidos). Entre estrategias: diferencas maiores que **14.57 pp** estao excluidas (15 pares; so as 4 originais: 10.84 pp).
- **Homburg · AST · fixo** (DP entre sementes 1.83 pp). MSM vs controle (-0.47 pp): efeitos maiores que **3.73 pp** estao excluidos (ganhos acima de 2.79 pp e perdas acima de 3.73 pp excluidos). Entre estrategias: diferencas maiores que **5.21 pp** estao excluidas (7 pares; so as 4 originais: 4.66 pp).
- **FMA-small · AST · fixo** (DP entre sementes 1.36 pp). MSM vs controle (-1.97 pp): efeitos maiores que **4.57 pp** estao excluidos (ganhos acima de 0.64 pp e perdas acima de 4.57 pp excluidos). Entre estrategias: diferencas maiores que **5.40 pp** estao excluidas (15 pares; so as 4 originais: 4.79 pp).
- **Combinado (principal, k = 4)**. MSM vs controle (-0.03 pp): efeitos maiores que **2.15 pp** estao excluidos (ganhos acima de 2.09 pp e perdas acima de 2.15 pp excluidos; HKSJ). Entre as 4 estrategias originais: diferencas maiores que **5.02 pp** estao excluidas (IC simultaneo, 6 pares).
- **Combinado, so AST (k = 3)**. MSM vs controle (-0.59 pp): efeitos maiores que **4.16 pp** estao excluidos (ganhos acima de 2.98 pp e perdas acima de 4.16 pp excluidos; HKSJ). Entre as 4 estrategias originais: diferencas maiores que **10.02 pp** estao excluidas (IC simultaneo, 6 pares).

DP tipico entre sementes (mediana dos recortes): **1.54 pp**.
