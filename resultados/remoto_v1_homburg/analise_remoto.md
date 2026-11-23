# Analise da execucao remota (FMA-small e Homburg)

Gerado por `analisar_remoto.py`. A regra metodologica do projeto e que
**nenhum resultado e considerado estabelecido antes de ser replicado com
o dobro das sementes** -- ela ja derrubou cinco resultados aparentes, e
acertou uma previsao. Leia as secoes abaixo com isso em mente.

Ambiente: NVIDIA GeForce RTX 5060 | torch 2.14.1+cu130 | Windows-11-10.0.26200-SP0

## HOMBURG

Sementes completas: **6** (42, 43, 44, 45, 46, 47)

### Acuracia por condicao

| condicao | media | desvio-padrao | IC 95% da media | min | max |
|---|---|---|---|---|---|
| controle | 0.8088 | 0.0318 | [0.775; 0.842] | 0.7474 | 0.8351 |
| random | 0.8053 | 0.0199 | [0.784; 0.826] | 0.7754 | 0.8281 |
| spectral_energy | 0.8018 | 0.0103 | [0.791; 0.813] | 0.7895 | 0.8140 |
| onset | 0.8018 | 0.0119 | [0.789; 0.814] | 0.7860 | 0.8175 |
| combined | 0.8076 | 0.0156 | [0.791; 0.824] | 0.7860 | 0.8316 |

### Poder estatistico

- Conjunto de teste: **285** exemplos (um acerto = 0.35 pp)
- Menor diferenca detectavel (alpha 0,05, poder 0,80): **8.4 pp**
- Referencia do GTZAN: 150 exemplos, MDE 10,4 pp
- Este dataset resolve diferencas **menores** que o GTZAN.

### As estrategias se separam? (Friedman, blocos = sementes)

- chi2 = 1.603, **p = 0.6586** (6 sementes)
- NAO ha diferenca detectavel entre as estrategias (mesmo desfecho do GTZAN).

#### Pares de estrategias (t pareado por semente)

| par | delta medio (pp) | p |
|---|---|---|
| onset vs combined | -0.58 | 0.3927 |
| spectral_energy vs combined | -0.58 | 0.5713 |
| random vs spectral_energy | +0.35 | 0.6436 |
| random vs onset | +0.35 | 0.7234 |
| random vs combined | -0.23 | 0.8033 |
| spectral_energy vs onset | +0.00 | 1.0000 |

### O MSM ajuda? (cada estrategia contra o controle)

| estrategia | delta medio (pp) | positivo em | p (t pareado) |
|---|---|---|---|
| random | -0.35 | 3 de 6 | 0.7607 |
| spectral_energy | -0.70 | 1 de 6 | 0.5457 |
| onset | -0.70 | 2 de 6 | 0.6835 |
| combined | -0.12 | 3 de 6 | 0.9436 |

Agregado das 24 comparacoes (4 estrategias x 6 sementes): **-0.47 pp**, positivo em 9 de 24.

Teste **conservador** (media por semente, n = 6, porque as 4 estrategias de uma semente dividem o mesmo controle e nao sao independentes): **p = 0.7274**.

#### McNemar exato, estrategia vs controle, exemplo a exemplo

Teste pareado de verdade (usa a concordancia por exemplo, nao so as duas acuracias). `so_estrategia` = exemplos que so a estrategia acertou.

| semente | estrategia | so_estrategia | so_controle | p |
|---|---|---|---|---|
| 42 | random | 28 | 20 | 0.3123 |
| 42 | spectral_energy | 26 | 14 | 0.0807 |
| 42 | onset | 33 | 14 | 0.0079 |
| 42 | combined | 33 | 16 | 0.0213 |
| 43 | random | 19 | 14 | 0.4869 |
| 43 | spectral_energy | 16 | 21 | 0.5114 |
| 43 | onset | 16 | 14 | 0.8555 |
| 43 | combined | 17 | 11 | 0.3449 |
| 44 | random | 10 | 15 | 0.4244 |
| 44 | spectral_energy | 10 | 15 | 0.4244 |
| 44 | onset | 14 | 24 | 0.1433 |
| 44 | combined | 10 | 23 | 0.0351 |
| 45 | random | 9 | 22 | 0.0294 |
| 45 | spectral_energy | 11 | 21 | 0.1102 |
| 45 | onset | 11 | 22 | 0.0801 |
| 45 | combined | 10 | 20 | 0.0987 |
| 46 | random | 14 | 12 | 0.8450 |
| 46 | spectral_energy | 11 | 11 | 1.0000 |
| 46 | onset | 11 | 15 | 0.5572 |
| 46 | combined | 12 | 15 | 0.7011 |
| 47 | random | 15 | 18 | 0.7283 |
| 47 | spectral_energy | 10 | 14 | 0.5413 |
| 47 | onset | 13 | 21 | 0.2295 |
| 47 | combined | 16 | 15 | 1.0000 |

4 de 24 comparacoes significativas a 5% -- mas com 24 testes, ~1.2 falsos positivos sao esperados por acaso.

### O pretexto MSM aprendeu alguma coisa?

Queda da perda de reconstrucao na validacao, do inicio ao fim do Estagio 1: media **19.0%** (min 14.3%, max 28.3%, n = 24).

Isso importa para a interpretacao: se a perda NAO cai, um resultado nulo pode significar apenas que o pre-treino nao rodou de fato. Se cai, o nulo e sobre o metodo.

### Leitura

**6 sementes completas**, mesmo numero do AST no GTZAN. Comparavel diretamente com o resultado principal do trabalho.

Custo: 16.7 h de GPU em 30 execucoes (mediana 37 min cada).

## Referencia: o que o GTZAN ja dizia

| | AST no GTZAN (6 sementes) | CNN no GTZAN (10 sementes) |
|---|---|---|
| MSM vs controle | +0,81 pp, p conservador = 0,481 | -0,70 pp, p = 0,111 |
| Friedman entre estrategias | p = 0,309 | p = 0,315 |
| Conjunto de teste | 150 exemplos, MDE 10,4 pp | 150 exemplos |

As duas arquiteturas concordam: no GTZAN o MSM nao produz efeito detectavel e as estrategias nao se separam. A pergunta que esta corrida responde e se isso **generaliza** para outros datasets e taxonomias.
