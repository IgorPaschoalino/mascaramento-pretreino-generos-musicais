#!/usr/bin/env bash
# Regrava RESULTADOS_PARA_ENVIAR.zip com tudo o que ja terminou (Linux).
cd "$(dirname "$0")"
[ -x venv/bin/python ] || { echo "[ERRO] Ambiente nao encontrado."; exit 1; }
venv/bin/python src/analisar_remoto.py > /dev/null 2>&1
venv/bin/python src/coletar_resultados.py
