#!/usr/bin/env bash
# PASSO 2 (Linux) -- Roda TUDO; pode ser interrompido e retomado. Para
# continuar, rode o MESMO comando de novo. Ver src/rodar_tudo.py.
#
#     bash rodar_tudo.sh                      # roda ate acabar
#     bash rodar_tudo.sh --parar-as 07:30     # pausa sozinho as 07:30
#     touch PAUSAR                            # (outro terminal) pausa limpa
cd "$(dirname "$0")"
[ -x venv/bin/python ] || { echo "[ERRO] Rode primeiro:  bash instalar.sh"; exit 1; }
exec venv/bin/python src/rodar_tudo.py "$@"
