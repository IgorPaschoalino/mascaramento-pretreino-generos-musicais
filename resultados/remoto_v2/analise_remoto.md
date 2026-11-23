# Analise da execucao remota (FMA-small e Homburg)

Gerado por `analisar_remoto.py`. A regra metodologica do projeto e que
**nenhum resultado e considerado estabelecido antes de ser replicado com
o dobro das sementes** -- ela ja derrubou cinco resultados aparentes, e
acertou uma previsao. Leia as secoes abaixo com isso em mente.

Ambiente: NVIDIA GeForce RTX 5090 | torch 2.14.1+cu130 | Windows-11-10.0.26200-SP0

## FMA

Sementes completas: **6** (42, 43, 44, 45, 46, 47)

### Acuracia por condicao

| condicao | media | desvio-padrao | IC 95% da media | min | max |
|---|---|---|---|---|---|
| controle | 0.6140 | 0.0240 | [0.589; 0.639] | 0.5687 | 0.6312 |
| random | 0.5948 | 0.0132 | [0.581; 0.609] | 0.5713 | 0.6112 |
| spectral_energy | 0.5858 | 0.0057 | [0.580; 0.592] | 0.5787 | 0.5913 |
| onset | 0.6090 | 0.0126 | [0.596; 0.622] | 0.5938 | 0.6250 |
| combined | 0.5938 | 0.0099 | [0.583; 0.604] | 0.5813 | 0.6050 |
| inverse_energy | 0.5963 | 0.0100 | [0.586; 0.607] | 0.5813 | 0.6088 |
| specaugment | 0.5863 | 0.0126 | [0.573; 0.599] | 0.5737 | 0.6075 |

### Poder estatistico

- Conjunto de teste: **800** exemplos (um acerto = 0.12 pp)
- Menor diferenca detectavel (alpha 0,05, poder 0,80): **6.7 pp**
- Referencia do GTZAN: 150 exemplos, MDE 10,4 pp
- Este dataset resolve diferencas **menores** que o GTZAN.

### As estrategias se separam? (Friedman, blocos = sementes)

- todas as 6 estrategias: chi2 = 12.225, **p = 0.0318** (6 sementes) -- ha diferenca detectavel
- so as 4 originais: chi2 = 8.085, **p = 0.0443** (6 sementes) -- ha diferenca detectavel

#### Pares de estrategias (t pareado por semente)

| par | delta medio (pp) | p |
|---|---|---|
| onset vs specaugment | +2.27 | 0.0066 |
| spectral_energy vs onset | -2.31 | 0.0110 |
| random vs onset | -1.42 | 0.0647 |
| onset vs combined | +1.52 | 0.0683 |
| inverse_energy vs specaugment | +1.00 | 0.0772 |
| spectral_energy vs inverse_energy | -1.04 | 0.1006 |
| onset vs inverse_energy | +1.27 | 0.1162 |
| spectral_energy vs combined | -0.79 | 0.2037 |
| random vs specaugment | +0.85 | 0.2154 |
| random vs spectral_energy | +0.90 | 0.2309 |
| combined vs specaugment | +0.75 | 0.3118 |
| combined vs inverse_energy | -0.25 | 0.6003 |
| random vs inverse_energy | -0.15 | 0.8587 |
| random vs combined | +0.10 | 0.9085 |
| spectral_energy vs specaugment | -0.04 | 0.9486 |

### O MSM ajuda? (cada estrategia contra o controle)

| estrategia | delta medio (pp) | positivo em | p (t pareado) |
|---|---|---|---|
| random | -1.92 | 1 de 6 | 0.1802 |
| spectral_energy | -2.81 | 1 de 6 | 0.0167 |
| onset | -0.50 | 1 de 6 | 0.6939 |
| combined | -2.02 | 1 de 6 | 0.1520 |
| inverse_energy | -1.77 | 2 de 6 | 0.1481 |
| specaugment | -2.77 | 1 de 6 | 0.0422 |

Agregado das 36 comparacoes (estrategias x 6 sementes): **-1.97 pp**, positivo em 7 de 36.

Teste **conservador** (media por semente, n = 6, porque as 4 estrategias de uma semente dividem o mesmo controle e nao sao independentes): **p = 0.1104**.

#### McNemar exato, estrategia vs controle, exemplo a exemplo

Teste pareado de verdade (usa a concordancia por exemplo, nao so as duas acuracias). `so_estrategia` = exemplos que so a estrategia acertou.

| semente | estrategia | so_estrategia | so_controle | p |
|---|---|---|---|---|
| 42 | random | 61 | 106 | 0.0006 |
| 42 | spectral_energy | 64 | 93 | 0.0251 |
| 42 | onset | 62 | 82 | 0.1130 |
| 42 | combined | 70 | 88 | 0.1760 |
| 42 | inverse_energy | 59 | 86 | 0.0305 |
| 42 | specaugment | 58 | 101 | 0.0008 |
| 43 | random | 77 | 86 | 0.5310 |
| 43 | spectral_energy | 66 | 87 | 0.1056 |
| 43 | onset | 71 | 80 | 0.5152 |
| 43 | combined | 80 | 84 | 0.8149 |
| 43 | inverse_energy | 73 | 72 | 1.0000 |
| 43 | specaugment | 64 | 75 | 0.3964 |
| 44 | random | 63 | 87 | 0.0600 |
| 44 | spectral_energy | 72 | 100 | 0.0392 |
| 44 | onset | 63 | 85 | 0.0840 |
| 44 | combined | 59 | 95 | 0.0046 |
| 44 | inverse_energy | 57 | 82 | 0.0414 |
| 44 | specaugment | 65 | 106 | 0.0021 |
| 45 | random | 73 | 84 | 0.4249 |
| 45 | spectral_energy | 69 | 99 | 0.0250 |
| 45 | onset | 68 | 78 | 0.4565 |
| 45 | combined | 60 | 94 | 0.0076 |
| 45 | inverse_energy | 66 | 101 | 0.0083 |
| 45 | specaugment | 59 | 88 | 0.0206 |
| 46 | random | 99 | 73 | 0.0563 |
| 46 | spectral_energy | 55 | 47 | 0.4884 |
| 46 | onset | 101 | 59 | 0.0011 |
| 46 | combined | 99 | 74 | 0.0677 |
| 46 | inverse_energy | 90 | 71 | 0.1558 |
| 46 | specaugment | 52 | 42 | 0.3533 |
| 47 | random | 77 | 106 | 0.0382 |
| 47 | spectral_energy | 56 | 91 | 0.0049 |
| 47 | onset | 73 | 78 | 0.7449 |
| 47 | combined | 62 | 92 | 0.0192 |
| 47 | inverse_energy | 67 | 85 | 0.1677 |
| 47 | specaugment | 69 | 88 | 0.1506 |

16 de 36 comparacoes significativas a 5% -- mas com 36 testes, ~1.8 falsos positivos sao esperados por acaso.

### O pretexto MSM aprendeu alguma coisa?

Queda da perda de reconstrucao na validacao, do inicio ao fim do Estagio 1: media **24.4%** (min 18.0%, max 39.9%, n = 36).

Isso importa para a interpretacao: se a perda NAO cai, um resultado nulo pode significar apenas que o pre-treino nao rodou de fato. Se cai, o nulo e sobre o metodo.

### Leitura

**6 sementes completas**, mesmo numero do AST no GTZAN. Comparavel diretamente com o resultado principal do trabalho.

Custo: 115.2 h de GPU em 42 execucoes (mediana 196 min cada).

## HOMBURG

Sementes completas: **6** (42, 43, 44, 45, 46, 47)

Sem controle nesta corrida (reaproveitado da corrida anterior): so a comparacao entre as estrategias presentes.

### Acuracia por condicao

| condicao | media | desvio-padrao | IC 95% da media | min | max |
|---|---|---|---|---|---|
| inverse_energy | 0.8158 | 0.0096 | [0.806; 0.826] | 0.8000 | 0.8281 |
| specaugment | 0.8035 | 0.0188 | [0.784; 0.823] | 0.7825 | 0.8246 |

### Poder estatistico

- Conjunto de teste: **285** exemplos (um acerto = 0.35 pp)
- Menor diferenca detectavel (alpha 0,05, poder 0,80): **8.2 pp**
- Referencia do GTZAN: 150 exemplos, MDE 10,4 pp
- Este dataset resolve diferencas **menores** que o GTZAN.

### As estrategias se separam? (Friedman, blocos = sementes)

- Menos de 3 estrategias nesta corrida -- ver os pares abaixo.

#### Pares de estrategias (t pareado por semente)

| par | delta medio (pp) | p |
|---|---|---|
| inverse_energy vs specaugment | +1.23 | 0.2348 |

### O pretexto MSM aprendeu alguma coisa?

Queda da perda de reconstrucao na validacao, do inicio ao fim do Estagio 1: media **17.7%** (min 16.4%, max 18.5%, n = 12).

Isso importa para a interpretacao: se a perda NAO cai, um resultado nulo pode significar apenas que o pre-treino nao rodou de fato. Se cai, o nulo e sobre o metodo.

### Leitura

**6 sementes completas**, mesmo numero do AST no GTZAN. Comparavel diretamente com o resultado principal do trabalho.

Custo: 8.5 h de GPU em 12 execucoes (mediana 42 min cada).

## Referencia: o que o GTZAN ja dizia

| | AST no GTZAN (6 sementes) | CNN no GTZAN (10 sementes) |
|---|---|---|
| MSM vs controle | +0,81 pp, p conservador = 0,481 | -0,70 pp, p = 0,111 |
| Friedman entre estrategias | p = 0,309 | p = 0,315 |
| Conjunto de teste | 150 exemplos, MDE 10,4 pp | 150 exemplos |

As duas arquiteturas concordam: no GTZAN o MSM nao produz efeito detectavel e as estrategias nao se separam. A pergunta que esta corrida responde e se isso **generaliza** para outros datasets e taxonomias.
