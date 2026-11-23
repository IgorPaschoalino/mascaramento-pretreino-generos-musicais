"""
coletar_resultados.py
=====================
Empacota SO os resultados num zip pequeno, para devolver a quem enviou o pacote.

    python coletar_resultados.py

Inclui: metricas por execucao, predicoes por exemplo (necessarias para os testes
pareados), historico de perda do MSM, logs, impressao digital do ambiente e a
analise gerada.

Exclui: caches de dataset (~5 GB), checkpoints e encoders MSM. Nada disso e
necessario para a analise e nao caberia num anexo.

O zip resultante costuma ter poucos MB -- pode ir por e-mail ou mensagem.

v2: o nome e FIXO (RESULTADOS_PARA_ENVIAR.zip) e o run_remoto.py o regrava ao
fim de cada bloco de semente -- ha sempre um unico arquivo, sempre atualizado.
"""

import os
import sys
import json
import zipfile
import datetime

RESULTS_DIR = 'resultados'
# Extensoes que interessam. Qualquer .pt/.safetensors/.arrow que tenha sobrado
# em resultados/ fica de fora por nao estar nesta lista.
EXTENSOES = {'.json', '.md', '.log', '.csv', '.txt'}
LIMITE_AVISO_MB = 50
SAIDA = 'RESULTADOS_PARA_ENVIAR.zip'


def coletar(silencioso=False):
    if not os.path.isdir(RESULTS_DIR):
        print('Pasta %s nao existe -- nada foi executado ainda.' % RESULTS_DIR)
        sys.exit(1)

    arquivos = []
    for raiz, _, nomes in os.walk(RESULTS_DIR):
        for nome in nomes:
            caminho = os.path.join(raiz, nome)
            if os.path.splitext(nome)[1].lower() in EXTENSOES:
                arquivos.append(caminho)

    if not arquivos:
        print('Nenhum resultado encontrado em %s.' % RESULTS_DIR)
        sys.exit(1)

    # Resumo do que esta sendo enviado, para que o destinatario saiba de imediato
    # quanto da corrida foi concluido sem ter que abrir nada.
    resumo = {'gerado_em': datetime.datetime.now().isoformat(timespec='seconds')}
    res_file = os.path.join(RESULTS_DIR, 'resultados_remoto.json')
    sem_execucoes = not os.path.exists(res_file)
    if sem_execucoes:
        # Nao e erro: se o teste de fumaca falhou, empacotar so os logs e
        # exatamente o que precisa ser enviado para diagnostico. Mas nao pode
        # parecer que a corrida produziu resultados.
        resumo['aviso'] = ('Nenhuma execucao de treino concluida -- este pacote '
                           'contem apenas logs, para diagnostico.')
        resumo['execucoes_totais'] = 0
    if os.path.exists(res_file):
        with open(res_file, encoding='utf-8') as f:
            r = json.load(f)
        por_dataset = {}
        for k, v in r.items():
            d = por_dataset.setdefault(v['dataset'], {'execucoes': 0, 'sementes': set(),
                                                      'horas': 0.0})
            d['execucoes'] += 1
            d['sementes'].add(v['semente'])
            d['horas'] += v.get('minutos', 0) / 60.0
        resumo['execucoes_totais'] = len(r)
        resumo['por_dataset'] = {
            k: {'execucoes': v['execucoes'],
                'sementes': sorted(v['sementes']),
                # 7 condicoes por bloco (controle + 6 estrategias); no Homburg
                # reaproveitado da v1 o bloco novo tem so as 2 estrategias novas.
                'execucoes_por_semente': {
                    str(s): sum(1 for vv in r.values()
                                if vv['dataset'] == k and vv['semente'] == s)
                    for s in sorted(v['sementes'])},
                'horas_gpu': round(v['horas'], 1)}
            for k, v in por_dataset.items()}

    caminho_resumo = os.path.join(RESULTS_DIR, 'RESUMO_ENVIO.json')
    with open(caminho_resumo, 'w', encoding='utf-8') as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    if caminho_resumo not in arquivos:
        arquivos.append(caminho_resumo)

    saida = SAIDA
    tmp = saida + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for caminho in sorted(arquivos):
            z.write(caminho, os.path.relpath(caminho, '.'))
    os.replace(tmp, saida)          # nunca deixa um zip pela metade

    mb = os.path.getsize(saida) / (1024 ** 2)
    if silencioso:
        return saida
    print()
    print('=' * 66)
    if sem_execucoes:
        print('PACOTE DE DIAGNOSTICO PRONTO (nenhum treino concluido)')
    else:
        print('PACOTE DE RESULTADOS PRONTO')
    print('=' * 66)
    print()
    if sem_execucoes:
        print('  ATENCAO: nenhuma execucao de treino foi concluida ainda, entao')
        print('  este zip contem apenas os logs. Serve para diagnostico se algo')
        print('  deu errado -- pode enviar, mas nao ha resultados nele.')
        print()
    print('  Arquivo : %s' % saida)
    print('  Tamanho : %.1f MB' % mb)
    print('  Conteudo: %d arquivos' % len(arquivos))
    print()
    if 'por_dataset' in resumo:
        print('  Progresso:')
        for d, v in sorted(resumo['por_dataset'].items()):
            print('    %-9s %2d execucoes | sementes %s | %.1f h de GPU'
                  % (d, v['execucoes'], v['sementes'], v['horas_gpu']))
        print()
    if mb > LIMITE_AVISO_MB:
        print('  AVISO: %.0f MB pode nao passar por e-mail. Use um link '
              '(Drive, WeTransfer).' % mb)
        print()
    print('  Envie este arquivo para quem lhe passou o pacote.')
    print('  Os caches de dataset e os checkpoints NAO vao nele, de proposito:')
    print('  sao alguns GB e nao servem para a analise.')
    print()
    print('  Pode continuar a corrida depois de enviar -- o zip e')
    print('  regravado com tudo o que existir sempre que um bloco termina.')
    print('=' * 66)


if __name__ == '__main__':
    coletar()
