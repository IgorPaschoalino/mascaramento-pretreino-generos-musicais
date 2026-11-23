# Um Estudo Comparativo de Estratégias de Mascaramento no Pré-treino Auto-supervisionado para Classificação de Gêneros Musicais

Igor Silva de P. Paschoalino e Matheus Barros Pereira (orientador)
Pontifícia Universidade Católica de Minas Gerais (PUC Minas), 2026.

Este repositório tem o código e os resultados do Trabalho de Conclusão de Curso.

O protocolo tem dois estágios. Primeiro, um pré-treino auto-supervisionado reconstrói patches mascarados do espectrograma Mel (MSM). Depois, um ajuste fino limpo faz a classificação de gênero.

São comparadas seis estratégias de mascaramento e um controle sem pré-treino:

| Nome no código | Nome no artigo |
|---|---|
| `random` | aleatória |
| `spectral_energy` | energia |
| `onset` | onset |
| `combined` | combinada |
| `inverse_energy` | energia inversa |
| `specaugment` | faixas |

Os modelos são dois: o AST, pré-treinado no AudioSet, e uma CNN treinada do zero. As bases são GTZAN, Homburg e FMA-small.

No total são 581 execuções. O resultado é nulo nas duas perguntas: nenhuma estratégia se distingue das outras, e o pré-treino não supera o controle. Os intervalos de confiança limitam o tamanho dos efeitos que ainda seriam compatíveis com os dados.

## Estrutura

```
masking/               pontuação dos patches e seleção da máscara (as 6 estratégias)
preprocessing/         áudio -> espectrograma Mel -> patches 16x16 (GTZAN)
v2_common.py           AST: modelo de reconstrução (MSM) e utilitários
cnn_common.py          CNN: modelo de reconstrução e classificador
cnn_data.py            CNN: carga dos dados e máscaras
run_v2_experiment.py   treino AST no GTZAN, divisão fixa
run_cnn_experiment.py  treino CNN no GTZAN, divisão fixa
run_kfold_gtzan.py     treino AST ou CNN no GTZAN com 5 folds
remoto/                pacote que rodou Homburg e FMA-small em outra máquina (Linux)
analysis_ciclo2.py     carga dos resultados (módulo usado por analysis_final.py)
analysis_final.py      testes, Holm e meta-análise -> analysis_final.md
figura_forest.py       Figura de florestas -> figuras/figura_forest.pdf/.png
figura_mascaras.py     Figura das máscaras -> figuras/figura_mascaras.pdf/.png
resultados/            resultados das 581 execuções (JSON)
figuras/               figuras do artigo
analysis_final.md      relatório gerado por analysis_final.py
```

Em `resultados/` ficam:
- `results_v2.json`: AST na divisão fixa;
- `results_cnn.json`: CNN na divisão fixa;
- `results_kfold.json`: 5 folds, com as predições;
- `remoto_v1_homburg/`: Homburg, controle e as 4 primeiras estratégias, numa RTX 5060;
- `remoto_v2/`: FMA-small e as 2 estratégias novas no Homburg, numa RTX 5090.

No Homburg, o controle e as 4 primeiras estratégias rodaram com a versão v1 do pacote remoto, e as 2 estratégias de referência com a v2 (`remoto/`). A pasta `remoto_v1_homburg/` tem só os resultados, sem o código da v1.

Cada pasta `execucoes/` traz `result.json`, `test_preds.json` e `msm_history.json` de cada execução.

O `results_v2.json` também guarda 19 execuções de testes antigos (protocolo de um estágio e varredura de `mask_ratio`). O `analysis_ciclo2.py` as descarta, e elas não fazem parte das 581.

## Instalação

Python 3.13:

```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Para só refazer as tabelas e figuras, sem GPU, bastam três pacotes:

```bash
pip install numpy==2.4.4 scipy==1.17.1 matplotlib==3.10.9
```

## Refazer as tabelas e figuras (sem GPU)

Rode da raiz do repositório:

```bash
python analysis_final.py
python figura_forest.py
```

O `analysis_final.py` lê `resultados/` e reescreve o `analysis_final.md`. Esse relatório traz:
- Homburg separado por máquina;
- todos os testes de hipótese com correção de Holm;
- a meta-análise de efeitos aleatórios;
- os limites do tipo "efeitos maiores que X p.p. estão excluídos".

O `figura_forest.py` reescreve `figuras/figura_forest.pdf` e `.png` e imprime no terminal os valores de cada recorte. A figura tem dois painéis:
- (a) Estágio 1 menos controle, com o combinado em −0,03 p.p. [−2,15; +2,09];
- (b) seletivas (energia, onset, combinada) menos aleatória, com o combinado em +0,07 p.p. [−0,66; +0,80].

O AST com 5 folds tem só 1 semente e fica fora da figura; ele aparece no `analysis_final.md`.

O `figura_mascaras.py` precisa do cache pré-processado do GTZAN (ver abaixo), do `torch` e do `datasets`:

```bash
python figura_mascaras.py
```

Para analisar resultados de um novo treino, monte uma pasta com a mesma estrutura de `resultados/` e aponte para ela. Ex.: `TCC_RESULTADOS=meus_resultados python analysis_final.py`. No Windows PowerShell: `$env:TCC_RESULTADOS='meus_resultados'`.

## Re-treinar

Os áudios das bases não estão neste repositório, porque têm direitos de terceiros. É preciso uma GPU NVIDIA. Na RTX 2060 SUPER usada, uma execução AST leva cerca de 1 h e uma CNN alguns minutos.

### GTZAN (AST e CNN)

1. Baixe o GTZAN em <https://marsyas.info/downloads/datasets.html> e extraia para `data/gtzan/genres/<gênero>/*.wav`.
2. Gere os caches, um por estratégia com pontuação própria. `inverse_energy` usa o de `spectral_energy`, e `specaugment` usa o de `random`:

   ```bash
   python preprocessing/preprocessing.py --data-dir ./data/gtzan --mask-strategy random          --cache-dir ./cache
   python preprocessing/preprocessing.py --data-dir ./data/gtzan --mask-strategy spectral_energy --cache-dir ./cache
   python preprocessing/preprocessing.py --data-dir ./data/gtzan --mask-strategy onset           --cache-dir ./cache
   python preprocessing/preprocessing.py --data-dir ./data/gtzan --mask-strategy combined        --cache-dir ./cache
   ```

   Os caches ficam em `cache/gtzan-<estratégia>-p16-s42`, com a divisão fixa 699/150/150 e semente 42.
3. Treine:

   ```bash
   # AST, divisão fixa: controle + 6 estratégias, sementes 42-47
   python run_v2_experiment.py --only ctl --seeds 42 43 44 45 46 47
   python run_v2_experiment.py --only e2 --strategies random spectral_energy onset combined inverse_energy specaugment --seeds 42 43 44 45 46 47

   # CNN, divisão fixa: controle + 6 estratégias, sementes 42-51
   python run_cnn_experiment.py --strategies random spectral_energy onset combined inverse_energy specaugment --seeds 42 43 44 45 46 47 48 49 50 51

   # 5 folds: CNN com 10 sementes, AST com 1
   python run_kfold_gtzan.py --arch cnn --seeds 42 43 44 45 46 47 48 49 50 51
   python run_kfold_gtzan.py --arch ast --seeds 42
   ```

   Os scripts podem ser interrompidos e retomados: reinvocar pula o que já terminou. Os resultados vão para `results_v2.json`, `results_cnn.json` e `results_kfold.json` na raiz, e os modelos para `models_*/`.

   **Épocas de MSM no AST:** o padrão é 30 épocas. Nos resultados publicados, 10 das 36 execuções AST com pré-treino usaram 20: `random`, `spectral_energy` e `combined` nas sementes 42–44, e `onset` na semente 42. Para reproduzi-las exatamente, use `--msm-epochs 20` nessas execuções. O artigo discute essa diferença.

### Homburg e FMA-small (AST, pacote `remoto/`, Linux)

O código baixa o Homburg e o FMA-small sozinho, dos espelhos no Hugging Face Hub que foram usados nos experimentos:
- Homburg: [`mteb/music-genre`](https://huggingface.co/datasets/mteb/music-genre);
- FMA-small: [`rpmon/fma-genre-classification`](https://huggingface.co/datasets/rpmon/fma-genre-classification).

Fontes originais:
- FMA: Defferrard et al., <https://github.com/mdeff/fma>;
- Homburg: Homburg et al. (2005), *A Benchmark Dataset for Audio Classification and Clustering*, ISMIR.

```bash
cd remoto
bash instalar.sh                         # cria o venv e instala o torch compatível com o driver
bash rodar_tudo.sh --homburg-completo    # prepara os dados, faz um ensaio curto e roda tudo
```

A corrida pode ser pausada e retomada rodando o mesmo comando de novo.

Sem `--homburg-completo`, o pacote reaproveita o controle e as 4 primeiras estratégias do Homburg de uma corrida anterior e só roda as 2 novas. Foi o que aconteceu nos resultados publicados, por isso o Homburg está dividido em duas máquinas.

`remoto/src/masking/` é uma cópia idêntica de `masking/`, para o pacote rodar sozinho.

## Créditos

O mascaramento ponderado sem reposição e o *data collator* (`masking/patch_mask_collator.py`) foram adaptados de [ygorg/legal-masking](https://github.com/ygorg/legal-masking). O README desse repositório indica a licença CC BY 4.0. A adaptação levou o método do domínio de texto para patches de espectrograma, e as partes adaptadas mantêm a atribuição aos autores originais.

> Belfathi, A.; Gallina, Y.; Hernandez, N.; Monceaux, L.; Dufour, R. *Is Selective Masking A Key to Improving Domain Adaptation for Masked Language Model?* ICAIL 2025. doi:10.1145/3769126.3769216

```bibtex
@inproceedings{belfathi2025selective,
  author    = {Belfathi, Anas and Gallina, Ygor and Hernandez, Nicolas and Monceaux, Laura and Dufour, Richard},
  title     = {Is Selective Masking A Key to Improving Domain Adaptation for Masked Language Model?},
  booktitle = {ICAIL 2025},
  year      = {2025},
  doi       = {10.1145/3769126.3769216}
}
```

## Licença

MIT, veja `LICENSE`. As partes adaptadas do legal-masking seguem com a atribuição acima.
