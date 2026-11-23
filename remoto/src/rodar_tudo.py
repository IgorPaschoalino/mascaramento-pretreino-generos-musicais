"""
rodar_tudo.py
=============
Faz a corrida inteira, do inicio ao fim, em Windows ou Linux. Quem chama e o
RODAR.bat (Windows) ou o rodar_tudo.sh (Linux); nao precisa rodar a mao.

    venv\\Scripts\\python.exe src\\rodar_tudo.py                 (Windows)
    venv/bin/python src/rodar_tudo.py                          (Linux)
    ... src/rodar_tudo.py --parar-as 07:30                     (pausa sozinho)

Pode ser interrompido e retomado quantas vezes for preciso: rodar de novo pula
tudo o que ja terminou (o treino salva o progresso a cada epoca).

Etapas:
  1. testes de consistencia do codigo
  2. verificacao do ambiente (GPU, disco, internet)
  3. procura dados ja existentes na maquina (nao baixa de novo o que existe)
  4. preparo dos dados (Homburg e FMA-small)
  5. teste de ensaio em miniatura (prova que o treino funciona)
  6. CORRIDA (dias)
  7. analise e empacotamento

GARANTIA DE RESULTADO: aconteca o que acontecer (fim normal, pausa, Ctrl+C,
erro em qualquer etapa), ao sair este script regrava RESULTADOS_PARA_ENVIAR.zip
com tudo o que existir -- resultados e/ou logs. Toda a saida da tela tambem vai
para resultados/logs/rodar_tudo.log, que entra no zip; assim, se algo der
errado, o zip basta para diagnosticar.

No Windows, enquanto roda:
  - impede o computador de entrar em suspensao (ele volta ao normal ao sair);
  - desliga o "modo de edicao rapida" do console, que congela o programa
    quando alguem clica dentro da janela.
"""

import os
import sys
import time
import datetime
import argparse
import subprocess

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
os.chdir(RAIZ)
sys.path.insert(0, AQUI)

PY = sys.executable
WINDOWS = os.name == 'nt'
LOG_DIR = os.path.join('resultados', 'logs')
LOG = os.path.join(LOG_DIR, 'rodar_tudo.log')
ARQUIVO_PAUSA = 'PAUSAR'
CONTINUAR = 'RODAR.bat' if WINDOWS else 'bash rodar_tudo.sh'

os.makedirs(LOG_DIR, exist_ok=True)
_log = open(LOG, 'ab')


def escreve(texto):
    dados = texto.encode('utf-8', 'replace')
    try:
        sys.stdout.buffer.write(dados)
        sys.stdout.flush()
    except Exception:
        pass
    _log.write(dados)
    _log.flush()


def say(msg=''):
    escreve(msg + '\n')


def etapa(titulo):
    say()
    say('=' * 66)
    say(' %s   [%s]' % (titulo, time.strftime('%d/%m %H:%M')))
    say('=' * 66)


# ---------------------------------------------------------------------------
# Windows: nao suspender, nao congelar ao clicar na janela
# ---------------------------------------------------------------------------

def preparar_windows():
    if not WINDOWS:
        return
    import ctypes
    k = ctypes.windll.kernel32
    try:
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: o sistema nao suspende enquanto
        # este processo viver. A tela ainda pode apagar (economiza energia).
        k.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    try:
        h = k.GetStdHandle(-10)                      # STD_INPUT_HANDLE
        modo = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(modo)):
            # tira ENABLE_QUICK_EDIT_MODE (0x40), liga ENABLE_EXTENDED_FLAGS (0x80)
            k.SetConsoleMode(h, (modo.value & ~0x40) | 0x80)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Execucao das etapas, com a saida espelhada no log
# ---------------------------------------------------------------------------

class Interrompido(Exception):
    pass


def rodar(argv):
    """Roda `python argv...`, mostrando e gravando a saida. Devolve o codigo."""
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    p = subprocess.Popen([PY] + argv, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    interrompido = False
    while True:
        try:
            bloco = p.stdout.read1(4096) if hasattr(p.stdout, 'read1') else p.stdout.read(1)
            if not bloco:
                break
            try:
                sys.stdout.buffer.write(bloco)
                sys.stdout.flush()
            except Exception:
                pass
            _log.write(bloco)
            _log.flush()
        except KeyboardInterrupt:
            # O Ctrl+C chega tambem ao processo filho, que salva e sai sozinho;
            # aqui so esperamos ele terminar, para nao deixar nada pela metade.
            interrompido = True
    while True:
        try:
            code = p.wait()
            break
        except KeyboardInterrupt:
            interrompido = True
    if interrompido:
        raise Interrompido()
    return code


def quieto(argv):
    """Roda sem mostrar na tela (so no log). Nunca levanta excecao."""
    try:
        with open(LOG, 'ab') as f:
            return subprocess.call([PY] + argv, stdout=f, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
    except Exception:
        return -1


def empacotar():
    """Analise + zip. Chamado SEMPRE ao sair. Nunca levanta excecao."""
    quieto([os.path.join('src', 'analisar_remoto.py')])
    try:
        import coletar_resultados
        coletar_resultados.coletar(silencioso=True)
        return coletar_resultados.SAIDA
    except BaseException:
        return None


# ---------------------------------------------------------------------------
# Pausa entre etapas (o PAUSAR.bat / --parar-as tambem valem fora do treino)
# ---------------------------------------------------------------------------

PARAR_EM = {'quando': None}


def configurar_parada(hhmm):
    if not hhmm:
        return
    h, m = (int(x) for x in hhmm.split(':'))
    agora = datetime.datetime.now()
    alvo = agora.replace(hour=h, minute=m, second=0, microsecond=0)
    if alvo <= agora:
        alvo += datetime.timedelta(days=1)
    PARAR_EM['quando'] = alvo


def pausa_pedida():
    if os.path.exists(ARQUIVO_PAUSA):
        try:
            os.remove(ARQUIVO_PAUSA)
        except OSError:
            pass
        return 'pedido de pausa (PAUSAR)'
    if PARAR_EM['quando'] and datetime.datetime.now() >= PARAR_EM['quando']:
        return 'horario de parada (%s)' % PARAR_EM['quando'].strftime('%H:%M')
    return None


class Parar(Exception):
    def __init__(self, codigo, mensagem):
        Exception.__init__(self, mensagem)
        self.codigo = codigo
        self.mensagem = mensagem


def checar_pausa():
    motivo = pausa_pedida()
    if motivo:
        raise Parar(0, 'PAUSADO entre etapas: %s.' % motivo)


def obrigatoria(argv, erro):
    code = rodar(argv)
    if code == 3:                     # run_remoto: pausa pedida no meio do ensaio
        raise Parar(0, 'PAUSADO (progresso salvo).')
    if code != 0:
        raise Parar(1, erro + ' (codigo %d)' % code)


# ---------------------------------------------------------------------------

def etapas(args, extra):
    s = lambda nome: os.path.join('src', nome)          # noqa: E731

    etapa('1/7  Testes de consistencia (segundos)')
    obrigatoria([s('testar_pacote.py')],
                'Os testes falharam. NAO continue -- envie o zip para quem mandou o pacote.')
    checar_pausa()

    etapa('2/7  Verificacao do ambiente (segundos)')
    obrigatoria([s('verificar_ambiente.py')], 'O ambiente tem problemas (veja acima).')
    checar_pausa()

    etapa('3/7  Procurando dados ja existentes nesta maquina')
    rodar([s('localizar_dados.py'), 'procurar'])
    checar_pausa()

    etapa('4/7  Preparo dos dados (so na primeira vez: ~30 min + download)')
    obrigatoria([s('preprocess_remoto.py'), '--dataset', 'homburg'],
                'Falha no preparo do Homburg. Geralmente e conexao ou disco; '
                'rode de novo, o que ja foi feito nao se perde.')
    obrigatoria([s('localizar_dados.py'), 'compat'], 'Falha ao conferir o Homburg.')
    checar_pausa()
    obrigatoria([s('preprocess_remoto.py'), '--dataset', 'fma'],
                'Falha no preparo do FMA-small. Geralmente e conexao ou disco; '
                'rode de novo, o que ja foi feito nao se perde.')
    checar_pausa()

    etapa('5/7  Teste de ensaio em miniatura (~15 min na primeira vez)')
    obrigatoria([s('run_remoto.py'), '--fumaca', '--datasets', 'homburg', 'fma'],
                'O teste de ensaio falhou. NAO inicie a corrida -- envie o zip.')
    checar_pausa()

    etapa('6/7  CORRIDA (dias; o progresso e salvo a cada epoca)')
    rodar([s('run_remoto.py'), '--plano'] + extra)
    say(' Pode deixar rodando. Para PAUSAR: de dois cliques em PAUSAR.bat'
        if WINDOWS else ' Pode deixar rodando. Para PAUSAR:  touch PAUSAR')
    say(' (termina a epoca atual, salva e sai em poucos minutos).')
    say(' Para continuar depois:  %s' % CONTINUAR)
    say()
    corrida = [s('run_remoto.py')] + extra
    if args.parar_as:
        corrida += ['--parar-as', args.parar_as]
    code = rodar(corrida)
    if code == 3:
        raise Parar(0, 'PAUSADO durante o treino (progresso salvo).')
    if code != 0:
        raise Parar(code, 'A corrida parou com codigo %d. Rode %s de novo para '
                          'tentar outra vez; se repetir, envie o zip.' % (code, CONTINUAR))

    etapa('7/7  Analise e empacotamento')
    rodar([s('run_remoto.py'), '--plano'] + extra)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parar-as', default=None, metavar='HH:MM')
    args, extra = p.parse_known_args()   # o resto vai para o run_remoto.py (testes)
    configurar_parada(args.parar_as)
    preparar_windows()

    say()
    say('#' * 66)
    say(' rodar_tudo  %s   (python %s)' % (time.strftime('%d/%m/%Y %H:%M'),
                                           sys.version.split()[0]))
    if args.parar_as:
        say(' Pausa automatica as %s.' % args.parar_as)
    say('#' * 66)

    codigo, mensagem, terminou = 0, None, False
    try:
        etapas(args, extra)
        terminou = True
    except Parar as pr:
        codigo, mensagem = pr.codigo, pr.mensagem
    except Interrompido:
        codigo, mensagem = 130, 'INTERROMPIDO com Ctrl+C (progresso salvo ate a ultima epoca).'
    except KeyboardInterrupt:
        codigo, mensagem = 130, 'INTERROMPIDO com Ctrl+C.'
    except BaseException as exc:          # qualquer outra coisa: registra e empacota
        import traceback
        say(traceback.format_exc())
        codigo, mensagem = 5, 'ERRO inesperado: %r' % (exc,)

    say()
    say('=' * 66)
    if terminou:
        say(' CORRIDA COMPLETA.')
    else:
        say(' %s' % mensagem)
    zipf = empacotar()
    if zipf and os.path.exists(zipf):
        kb = os.path.getsize(zipf) / 1024.0
        say(' Resultados ate aqui: %s  (%s)'
            % (os.path.abspath(zipf),
               '%.1f MB' % (kb / 1024) if kb >= 1024 else '%.0f KB' % kb))
        say(' Envie ESTE arquivo para quem mandou o pacote.')
    else:
        say(' [AVISO] Nao consegui gerar o zip. Envie a pasta resultados\\logs.'
            if WINDOWS else ' [AVISO] Nao consegui gerar o zip. Envie resultados/logs.')
    if not terminou:
        say(' Para continuar de onde parou:  %s' % CONTINUAR)
    say('=' * 66)
    return codigo


if __name__ == '__main__':
    sys.exit(main())
