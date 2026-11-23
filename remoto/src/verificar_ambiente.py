"""
verificar_ambiente.py
=====================
Conferencia do ambiente ANTES de comprometer dezenas de horas de GPU.

    python verificar_ambiente.py

Checa dependencias, GPU, espaco em disco e acesso ao HuggingFace, e imprime uma
estimativa de tempo calibrada pela placa que realmente esta na maquina.
"""

import os
import sys
import shutil
import platform

PROBLEMAS = []
AVISOS = []


def ok(msg):
    print('  [ok]     %s' % msg)


def falha(msg):
    print('  [FALHA]  %s' % msg)
    PROBLEMAS.append(msg)


def aviso(msg):
    print('  [aviso]  %s' % msg)
    AVISOS.append(msg)


print()
print('=' * 70)
print('VERIFICACAO DO AMBIENTE')
print('=' * 70)

# ---------------------------------------------------------------------------
print()
print('1. Python e bibliotecas')

if sys.version_info < (3, 9):
    falha('Python %d.%d e antigo demais (precisa 3.9+)' % sys.version_info[:2])
else:
    ok('Python %s' % sys.version.split()[0])

ESPERADAS = ['torch', 'transformers', 'datasets', 'numpy', 'librosa',
             'sklearn', 'scipy', 'soundfile', 'huggingface_hub']
faltando = []
for nome in ESPERADAS:
    try:
        mod = __import__(nome)
        ver = getattr(mod, '__version__', '?')
        ok('%-16s %s' % (nome, ver))
    except ImportError:
        falha('%-16s NAO INSTALADO' % nome)
        faltando.append(nome)

if faltando:
    print()
    print('  Instale com:  pip install -r requirements.txt')

# ---------------------------------------------------------------------------
print()
print('2. GPU')

velocidade_relativa = None
try:
    import torch
    if not torch.cuda.is_available():
        falha('CUDA nao disponivel -- o torch nao esta vendo a GPU.')
        print('           Verifique se instalou a versao com CUDA:')
        print('           INSTALAR.bat (Windows) ou bash instalar.sh (Linux) -- escolhe a CUDA certa')
    else:
        nome = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        ok('%s (%.1f GB de VRAM, CUDA %s)' % (nome, vram, torch.version.cuda))
        if vram < 6:
            falha('VRAM de %.1f GB e insuficiente. O experimento precisa de ~6 GB.' % vram)
        elif vram < 10:
            aviso('VRAM de %.1f GB e suficiente mas justa; nao use a maquina para '
                  'mais nada durante a corrida.' % vram)

        # Medicao rapida de throughput, para calibrar a estimativa de tempo pela
        # placa real em vez de um chute. Referencia: a RTX 2060 SUPER onde o
        # experimento do GTZAN rodou marca ~1.0.
        print()
        print('  Medindo velocidade da placa (alguns segundos)...')
        torch.cuda.synchronize()
        import time
        a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
        b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
        for _ in range(3):
            torch.mm(a, b)
        # Melhor de 3 rodadas, nao a media: o que interessa e o throughput de
        # pico da placa. Uma rodada lenta significa que outro processo pegou a
        # GPU no meio da medicao, e nao que a placa e mais devagar -- usar a
        # media faria a estimativa de tempo variar a cada execucao.
        n = 50
        melhor = float('inf')
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                torch.mm(a, b)
            torch.cuda.synchronize()
            melhor = min(melhor, time.time() - t0)
        tflops = (2 * 4096 ** 3 * n) / melhor / 1e12
        # TFLOPS fp16 MEDIDOS com este mesmo codigo na RTX 2060 SUPER onde todas
        # as 99 execucoes do GTZAN rodaram. E a base dos tempos de referencia
        # logo abaixo (56 min por execucao E2, 19 min por controle), entao as
        # duas coisas tem de vir da mesma placa.
        REF_2060S = 26.0
        velocidade_relativa = tflops / REF_2060S
        ok('%.0f TFLOPS (fp16) -- cerca de %.1fx a placa onde o GTZAN rodou'
           % (tflops, velocidade_relativa))
        del a, b
        torch.cuda.empty_cache()
except ImportError:
    falha('torch nao instalado -- impossivel checar a GPU.')

# ---------------------------------------------------------------------------
print()
print('3. Espaco em disco')

livre = shutil.disk_usage(os.path.abspath('.')).free / (1024 ** 3)
# FMA: ~7 GB de mp3 + ~8 GB do cache intermediario do `datasets` + 4 GB de
# espectrogramas; Homburg ~3 GB; checkpoints de trabalho ~3 GB. Se os dados ja
# existirem na maquina (localizar_dados.py), precisa bem menos.
MINIMO = 12.0
RECOMENDADO = 30.0
if livre < MINIMO:
    falha('Apenas %.1f GB livres. Sao necessarios pelo menos ~%.0f GB.' % (livre, MINIMO))
elif livre < RECOMENDADO:
    ok('%.1f GB livres' % livre)
    aviso('Se os dados ainda nao estiverem baixados, o ideal sao ~%.0f GB livres.'
          % RECOMENDADO)
else:
    ok('%.1f GB livres -- suficiente para tudo' % livre)

# ---------------------------------------------------------------------------
print()
print('4. Acesso ao HuggingFace (os dois datasets sao baixados de la)')

try:
    from huggingface_hub import HfApi
    api = HfApi()
    for repo in ('rpmon/fma-genre-classification', 'mteb/music-genre'):
        try:
            api.dataset_info(repo)
            ok('%s acessivel' % repo)
        except Exception as exc:
            falha('%s inacessivel: %s' % (repo, str(exc)[:90]))
except ImportError:
    falha('huggingface_hub nao instalado.')

# ---------------------------------------------------------------------------
print()
print('5. Sistema')

ok(platform.platform())
if platform.system() == 'Windows':
    print('           Nota: o pipeline foi desenvolvido no Windows e contorna um')
    print('           problema de DLL do torchcodec decodificando audio com')
    print('           librosa. Funciona igual no Linux.')

# ---------------------------------------------------------------------------
print()
print('=' * 70)
if PROBLEMAS:
    print('%d PROBLEMA(S) IMPEDEM A EXECUCAO:' % len(PROBLEMAS))
    for p in PROBLEMAS:
        print('  - %s' % p)
    print()
    print('Resolva antes de continuar. Em caso de duvida, avise quem enviou o pacote.')
    print('=' * 70)
    sys.exit(1)

if AVISOS:
    print('%d aviso(s), nao impedem a execucao:' % len(AVISOS))
    for a in AVISOS:
        print('  - %s' % a)
    print()

print('AMBIENTE OK.')

if velocidade_relativa:
    # Tempos medidos na placa de referencia (RTX 2060 SUPER), escalados pelo
    # tamanho do conjunto de treino e pela velocidade medida. E so uma ordem de
    # grandeza: o tempo real aparece na tela, por execucao, durante a corrida.
    # GTZAN: treino 699, ~56 min por execucao com MSM, ~19 min por controle.
    print()
    print('ESTIMATIVA PARA ESTA PLACA (ordem de grandeza)')
    print()
    esc_h = (1319 / 699.0) / velocidade_relativa
    esc_f = (6400 / 699.0) / velocidade_relativa
    homburg_h = 12 * 56 * esc_h / 60.0                 # 2 estrategias novas x 6
    homburg_full_h = 6 * (19 + 6 * 56) * esc_h / 60.0
    fma_semente_h = (19 + 6 * 56) * esc_f / 60.0
    print('  Homburg (2 estrategias novas x 6 sementes) ... %5.0f h' % homburg_h)
    print('     (se nao der para reaproveitar a corrida anterior: %.0f h)' % homburg_full_h)
    print('  FMA-small, por semente (7 condicoes) ........ %5.0f h' % fma_semente_h)
    print('  FMA-small, 6 sementes ....................... %5.0f h' % (6 * fma_semente_h))
    print('  ' + '-' * 58)
    print('  TUDO ........................................ %5.0f h (~%.1f dias)'
          % (homburg_h + 6 * fma_semente_h, (homburg_h + 6 * fma_semente_h) / 24))
    print()
    print('  NAO precisa ser seguido: pode pausar e retomar quantas vezes quiser.')
    print('  O progresso e salvo a cada epoca (poucos minutos).')

print()
print('=' * 70)
