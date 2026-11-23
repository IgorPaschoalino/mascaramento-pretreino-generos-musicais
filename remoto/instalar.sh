#!/usr/bin/env bash
# PASSO 1 -- Instalacao. Rode uma vez (rodar de novo e seguro: so completa o
# que faltar).
#
#     bash instalar.sh
#
# Cria um ambiente Python isolado em ./venv e instala o PyTorch com CUDA e as
# bibliotecas do requirements.txt. Nada e instalado fora desta pasta (alem do
# cache de download do pip).
set -uo pipefail
cd "$(dirname "$0")"

echo
echo "=========================================================="
echo " INSTALACAO"
echo "=========================================================="

# Python 3.10 a 3.13 (o 3.14 ainda nao tem todas as bibliotecas).
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" 2>/dev/null; then
        PY="$c"; break
    fi
done
if [ -z "$PY" ]; then
    echo "[ERRO] Python 3.10 a 3.13 nao encontrado."
    echo "       Ubuntu/Debian:  sudo apt install -y python3 python3-venv"
    exit 1
fi
echo "[1/4] Python: $("$PY" --version)"

if [ -x venv/bin/python ]; then
    echo "[2/4] Ambiente virtual ja existe, reaproveitando."
else
    echo "[2/4] Criando ambiente virtual..."
    if ! "$PY" -m venv venv; then
        echo "[ERRO] Falha ao criar o ambiente virtual. No Ubuntu/Debian instale:"
        echo "       sudo apt install -y python3-venv     (ou python3.X-venv)"
        rm -rf venv
        exit 1
    fi
fi
VPY=venv/bin/python
"$VPY" -m pip install --upgrade pip --quiet

gpu_ok () {
    "$VPY" - <<'EOF' 2>/dev/null
import torch, sys
if not torch.cuda.is_available():
    sys.exit(1)
a = torch.randn(256, 256, device='cuda', dtype=torch.float16)
float((a @ a).sum())
print('       torch %s | %s' % (torch.__version__, torch.cuda.get_device_name(0)))
EOF
}

if gpu_ok; then
    echo "[3/4] PyTorch com CUDA ja instalado e funcionando na GPU."
else
    echo "[3/4] Instalando o PyTorch com CUDA (~2,5 GB)..."
    # Da mais nova para a mais antiga: cu130 foi a que rodou na corrida
    # anterior (RTX 5060); as series RTX 50 exigem cu128 ou mais nova.
    # Se o driver for antigo demais para ele, tenta as versoes anteriores.
    OK=0
    for cu in cu130 cu128 cu126 cu124; do
        echo "       tentando $cu ..."
        # Sem desinstalar, o pip acharia que o torch da tentativa anterior ja
        # satisfaz o pedido e nao trocaria de versao de CUDA.
        "$VPY" -m pip uninstall -y torch >/dev/null 2>&1
        if "$VPY" -m pip install torch --index-url "https://download.pytorch.org/whl/$cu" \
           && gpu_ok; then
            OK=1; break
        fi
    done
    if [ "$OK" != 1 ]; then
        echo "[ERRO] O PyTorch instalou, mas nao conseguiu usar a GPU."
        echo "       Confira o driver da NVIDIA com:  nvidia-smi"
        exit 1
    fi
fi

echo "[4/4] Instalando as demais bibliotecas..."
if ! "$VPY" -m pip install -r requirements.txt; then
    echo "[ERRO] Falha ao instalar as dependencias. Rode de novo; se repetir,"
    echo "       envie a mensagem de erro acima."
    exit 1
fi

echo
echo "=========================================================="
echo " INSTALACAO CONCLUIDA"
echo "=========================================================="
echo " Agora rode:   bash rodar_tudo.sh"
echo
