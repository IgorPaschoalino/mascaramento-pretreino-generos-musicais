"""
localizar_dados.py
==================
Evita baixar e reprocessar o que ja existe na maquina.

    python src/localizar_dados.py procurar     # antes do preparo dos dados
    python src/localizar_dados.py compat       # depois do preparo

`procurar`:
  1. Se o cache processado (cache_remoto/<dataset>-p16) ja existe nesta pasta,
     nada a fazer.
  2. Senao, procura um cache_remoto/ de uma corrida ANTERIOR em outras pastas
     (vizinhas desta e dentro da pasta pessoal). Se achar um completo e com os
     mesmos parametros de espectrograma, grava o caminho em
     cache_remoto/onde_esta.json e ele e usado de la -- sem copiar nada.
  3. Informa se os audios originais ja estao no cache do HuggingFace. Se
     estiverem, o preparo reaproveita os arquivos e nao baixa de novo (isso e
     automatico; aqui so se mostra o que vai acontecer).

`compat`:
  Confere se o conjunto de TESTE do Homburg desta maquina e o mesmo da corrida
  anterior (01/10 e 05/10), comparando a sequencia dos 285 rotulos. Se for,
  grava resultados/compatibilidade.json e o run_remoto.py roda no Homburg so as
  2 estrategias novas, reaproveitando as 30 execucoes antigas. Se nao for, roda
  o Homburg completo -- misturar conjuntos de teste diferentes invalidaria a
  comparacao pareada.
"""

import os
import sys
import json

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

from remoto_common import DATASETS  # noqa: E402

MAPA = os.path.join('cache_remoto', 'onde_esta.json')
COMPAT = os.path.join('resultados', 'compatibilidade.json')
REF_HOMBURG = os.path.join(AQUI, 'referencia', 'homburg_v1.json')
PULAR = {'venv', '.venv', 'node_modules', '.git', '__pycache__', 'site-packages',
         'trabalho_tmp', 'proc', 'sys', 'dev',
         # Windows: pastas enormes do sistema, onde nunca ha um cache nosso.
         'AppData', 'Windows', 'Program Files', 'Program Files (x86)',
         'ProgramData', '$Recycle.Bin', 'System Volume Information'}

PARAMS_ESPERADOS = {
    'sample_rate': 16000, 'n_mels': 128, 'hop_length': 160, 'win_length': 400,
    'segment_secs': 10.0, 'patch_size': 16, 'fmin': 0, 'fmax': 8000,
}


def completo(d):
    return all(os.path.isfile(os.path.join(d, s, 'dataset_info.json'))
               for s in ('train', 'validation', 'test'))


def meta_ok(d, dataset):
    try:
        with open(os.path.join(d, 'meta.json'), encoding='utf-8') as f:
            m = json.load(f)
    except (OSError, ValueError):
        return False
    # Um cache de teste de fumaca (poucos exemplos) nao serve: exige o treino
    # inteiro (FMA 6400, Homburg 1319; folga para os mp3 que falham no decode).
    minimo = {'fma': 6000, 'homburg': 1250}[dataset]
    return (m.get('dataset') == dataset and m.get('seed') == 42
            and m.get('grid_shape') == [8, 62]
            and m.get('parametros_espectrograma') == PARAMS_ESPERADOS
            and m.get('splits', {}).get('train', 0) >= minimo)


def candidatos(nome, profundidade=5):
    """Caminhos .../cache_remoto/<nome> encontrados fora desta pasta."""
    raizes = []
    aqui = os.path.abspath('.')
    pai = os.path.dirname(aqui)
    extras = []
    if os.name == 'nt':
        # No Windows a pasta da corrida anterior pode estar em outro disco
        # (D:, E:...): a raiz de cada unidade entra, com a mesma profundidade.
        import string
        extras = [l + ':' + os.sep for l in string.ascii_uppercase
                  if os.path.isdir(l + ':' + os.sep)]
    for r in [pai, os.path.dirname(pai), os.path.expanduser('~')] + extras:
        if r and os.path.isdir(r) and r not in raizes:
            raizes.append(r)
    vistos = set()
    for raiz in raizes:
        base = raiz.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _ in os.walk(raiz):
            if dirpath.count(os.sep) - base >= profundidade:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames
                           if d not in PULAR and not (d.startswith('.') and d != '.')]
            if os.path.basename(dirpath) == 'cache_remoto':
                c = os.path.join(dirpath, nome)
                real = os.path.realpath(c)
                if (os.path.isdir(c) and real not in vistos
                        and os.path.realpath(dirpath) != os.path.realpath('cache_remoto')):
                    vistos.add(real)
                    yield c
                dirnames[:] = []


def hf_hub_dirs():
    dirs = []
    for env in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        if os.environ.get(env):
            dirs.append(os.environ[env])
    if os.environ.get('HF_HOME'):
        dirs.append(os.path.join(os.environ['HF_HOME'], 'hub'))
    dirs.append(os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub'))
    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]


def tamanho_gb(d):
    tot = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot / 1024 ** 3


REPOS = {'fma': 'datasets--rpmon--fma-genre-classification',
         'homburg': 'datasets--mteb--music-genre'}
TAMANHO_COMPLETO_GB = {'fma': 7.0, 'homburg': 1.7}


def procurar(datasets):
    os.makedirs('cache_remoto', exist_ok=True)
    mapa = {}
    if os.path.exists(MAPA):
        with open(MAPA, encoding='utf-8') as f:
            mapa = json.load(f)

    for ds in datasets:
        local = DATASETS[ds]['cache']
        nome = os.path.basename(local)
        print()
        print('[%s]' % ds)
        if completo(local):
            print('  Cache processado ja existe nesta pasta (%s). Nada a baixar.' % local)
            mapa.pop(ds, None)
            continue
        achado = None
        for c in candidatos(nome):
            if completo(c) and meta_ok(c, ds):
                achado = os.path.abspath(c)
                break
            print('  ignorado (incompleto ou com outros parametros): %s' % c)
        if achado:
            mapa[ds] = achado
            print('  Cache processado de uma corrida anterior ENCONTRADO:')
            print('    %s' % achado)
            print('  Sera usado de la -- sem baixar nem reprocessar.')
            continue
        mapa.pop(ds, None)
        brutos = [os.path.join(h, REPOS[ds]) for h in hf_hub_dirs()
                  if os.path.isdir(os.path.join(h, REPOS[ds]))]
        if brutos:
            gb = tamanho_gb(brutos[0])
            if gb >= TAMANHO_COMPLETO_GB[ds]:
                print('  Cache processado nao encontrado, mas os audios ja estao baixados')
                print('    (%s, %.1f GB).' % (brutos[0], gb))
                print('  O preparo vai reaproveita-los: so converte, quase nada a baixar.')
            else:
                print('  Audios PARCIALMENTE baixados (%.1f GB de ~%.1f GB) em %s.'
                      % (gb, TAMANHO_COMPLETO_GB[ds], brutos[0]))
                print('  O preparo reaproveita o que existe e baixa so o que falta.')
        else:
            print('  Nada encontrado nesta maquina: o preparo vai BAIXAR o dataset'
                  ' (%s).' % ('~7 GB' if ds == 'fma' else '~1,9 GB'))

    with open(MAPA, 'w', encoding='utf-8') as f:
        json.dump(mapa, f, indent=2, ensure_ascii=False)


def compat():
    from datasets import Dataset
    from remoto_common import cache_path_of

    os.makedirs('resultados', exist_ok=True)
    with open(REF_HOMBURG, encoding='utf-8') as f:
        ref = json.load(f)['y_true_teste']
    teste = os.path.join(cache_path_of('homburg'), 'test')
    if not os.path.isdir(teste):
        print('Cache do Homburg nao encontrado -- rode o preparo primeiro.')
        sys.exit(1)
    labels = [int(x) for x in Dataset.load_from_disk(teste)['label']]
    ok = labels == ref
    with open(COMPAT, 'w', encoding='utf-8') as f:
        json.dump({'homburg_v1_compativel': ok, 'n_teste': len(labels),
                   'n_teste_v1': len(ref)}, f, indent=2)
    print()
    if ok:
        print('Homburg: conjunto de teste IDENTICO ao da corrida anterior (%d exemplos).'
              % len(labels))
        print('  -> roda so as 2 estrategias novas; as 30 execucoes antigas sao reaproveitadas.')
    else:
        print('Homburg: conjunto de teste DIFERENTE da corrida anterior '
              '(%d vs %d exemplos, ou outra ordem).' % (len(labels), len(ref)))
        print('  -> roda o Homburg COMPLETO (controle + 6 estrategias), ~3,5x mais tempo.')


if __name__ == '__main__':
    modo = sys.argv[1] if len(sys.argv) > 1 else 'procurar'
    if modo == 'procurar':
        procurar(sys.argv[2:] or ['homburg', 'fma'])
    elif modo == 'compat':
        compat()
    else:
        print('uso: python src/localizar_dados.py [procurar|compat]')
        sys.exit(2)
