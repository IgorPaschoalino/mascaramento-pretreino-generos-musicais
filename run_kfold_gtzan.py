"""
run_kfold_gtzan.py
==================
Validacao cruzada estratificada em 5 folds no GTZAN, para AST ou CNN, com as
7 condicoes do ciclo 2 (controle + 4 estrategias originais + inverse_energy +
specaugment).

    venv\\Scripts\\python.exe run_kfold_gtzan.py --arch cnn --seeds 42 43 44 45 46
    venv\\Scripts\\python.exe run_kfold_gtzan.py --arch ast --seeds 42

Por que
-------
O limite do projeto e o tamanho do conjunto de teste: 150 exemplos no GTZAN,
MDE de 10,4 pp. Com 5 folds, cada semente avalia as 1000 musicas (cada uma
exatamente uma vez como teste), e as comparacoes pareadas passam a usar 1000
exemplos em vez de 150. O treino de cada fold tem ~680 exemplos, quase o
mesmo dos 699 do split fixo, entao o protocolo de treino nao muda.

Detalhes que precisam ir para o texto
-------------------------------------
- Folds fixos (StratifiedKFold, random_state=0), IGUAIS para todas as
  condicoes e sementes: a semente so muda o treino, nunca a divisao. A
  validacao de cada fold sao 15% do restante (estratificado, random_state=0).
- Os exemplos vem dos caches do split fixo, onde o treino teve crop
  aleatorio de 10 s e validacao/teste o crop inicial. No k-fold essas origens
  se misturam entre treino e teste. Isso vale igualmente para todas as
  condicoes, entao nao afeta a comparacao entre elas.
- Nao ha filtro de artista (o GTZAN tem repeticoes de artista conhecidas);
  vale o mesmo comentario: afeta o nivel absoluto, nao a comparacao.

Reaproveita SEM ALTERAR as funcoes de treino ja validadas:
`run_v2_experiment.msm_pretrain/finetune` (AST) e
`run_cnn_experiment.msm_pretrain/finetune` (CNN). Resumivel por execucao
(`experiment_state_kfold.json`), em ordem de bloco completo (semente, fold).
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import datetime

import numpy as np
import torch
from datasets import DatasetDict, concatenate_datasets
from sklearn.model_selection import StratifiedKFold, train_test_split

from masking.patch_mask_collator import cache_strategy_of

CONDICOES = ['noadapt', 'random', 'spectral_energy', 'onset', 'combined',
             'inverse_energy', 'specaugment']
N_FOLDS = 5
STATE_FILE = 'experiment_state_kfold.json'
RESULTS_FILE = 'results_kfold.json'
LOG_FILE = 'logs_kfold.log'
MIN_FREE_GB = 5.0
TESTE = {'on': False}      # --teste: 1 epoca por estagio, arquivos separados

log = logging.getLogger('kfold')


def setup_logging():
    fmt = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s',
                            '%d/%m %H:%M:%S')
    fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    for name in ('kfold', 'v2', 'cnn'):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.handlers = [fh, sh]
        lg.propagate = False


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return default


def save_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def free_gb():
    return shutil.disk_usage('.').free / 1024 ** 3


# ---------------------------------------------------------------------------
# Dados: os 1000 exemplos de cada cache, e os folds
# ---------------------------------------------------------------------------

_POOL = {}


def pool_of(cache_strategy):
    """Os 3 splits do cache concatenados (train, validation, test), em ordem."""
    if cache_strategy not in _POOL:
        d = DatasetDict.load_from_disk('cache/gtzan-%s-p16-s42' % cache_strategy)
        allds = concatenate_datasets([d['train'], d['validation'], d['test']])
        _POOL[cache_strategy] = allds.with_format('numpy')
    return _POOL[cache_strategy]


def checar_alinhamento():
    """Os 4 caches precisam ter os MESMOS exemplos na MESMA ordem, senao um
    fold seria uma divisao diferente para cada estrategia."""
    ref = pool_of('random')
    paths = list(ref['audio_path'])
    labels = np.asarray(ref['label'])
    for cs in ('spectral_energy', 'onset', 'combined'):
        p = pool_of(cs)
        if list(p['audio_path']) != paths or not np.array_equal(np.asarray(p['label']), labels):
            raise RuntimeError('cache %s desalinhado do random' % cs)
        for i in (0, 357, 998):
            if not np.array_equal(np.asarray(p[i]['input_values']),
                                  np.asarray(ref[i]['input_values'])):
                raise RuntimeError('patches diferentes entre caches no indice %d (%s)' % (i, cs))
    log.info('Caches alinhados: %d exemplos, mesmos arquivos e patches.', len(paths))
    return labels


def folds(labels):
    """[(train_idx, val_idx, test_idx)] fixos, independentes da semente."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    out = []
    for resto, teste in skf.split(np.zeros(len(labels)), labels):
        tr, va = train_test_split(resto, test_size=0.15, random_state=0,
                                  stratify=labels[resto])
        out.append((np.sort(tr), np.sort(va), np.sort(teste)))
    return out


def dataset_do_fold(cond, fold):
    """DatasetDict (formato numpy) do fold, a partir do cache da condicao."""
    cs = 'random' if cond == 'noadapt' else cache_strategy_of(cond)
    p = pool_of(cs)
    tr, va, te = fold
    return DatasetDict({'train': p.select(tr), 'validation': p.select(va),
                        'test': p.select(te)})


# ---------------------------------------------------------------------------
# Arquiteturas
# ---------------------------------------------------------------------------

def args_padrao(modulo):
    """Os hiperparametros-padrao do runner original (parse com argv vazio)."""
    antigo = sys.argv
    try:
        sys.argv = [antigo[0]]
        return modulo.arguments()
    finally:
        sys.argv = antigo


def rodar_ast(cond, fold, seed, tag):
    import run_v2_experiment as v2
    from v2_common import CleanCollator
    v2.OUT_DIR = 'models_kfold_ast'
    a = args_padrao(v2)
    a.seed, a.dataset, a.tag_suffix = seed, 'gtzan', ''
    if TESTE['on']:
        a.msm_epochs, a.ft_epochs = 1, 1
    ds = dataset_do_fold(cond, fold)
    clean = CleanCollator()
    enc = None
    hist = None
    if cond != 'noadapt':
        enc, hist = v2.msm_pretrain(cond, ds, a, tag)
    res = v2.finetune(ds, a, tag, collator=clean, eval_collator=clean,
                      dataset='gtzan', encoder_state=enc)
    if hist:
        res['msm_val_loss_first'] = hist[0]['msm_val_loss']
        res['msm_val_loss_last'] = hist[-1]['msm_val_loss']
    if enc and os.path.exists(enc):
        os.remove(enc)          # 340 MB por execucao, sem uso depois
    return res


def rodar_cnn(cond, fold, seed, tag):
    import run_cnn_experiment as cnn
    from cnn_data import TensorSplit
    cnn.OUT_DIR = 'models_kfold_cnn'
    a = args_padrao(cnn)
    a.seed = seed
    if TESTE['on']:
        a.msm_epochs, a.ft_epochs = 1, 1
    cnn.set_seed(seed)
    ds = dataset_do_fold(cond, fold)
    splits = {k: TensorSplit(ds[k]) for k in ('train', 'validation', 'test')}
    if cond == 'noadapt':
        return cnn.finetune(splits, a, tag, encoder_state=None)
    enc, hist = cnn.msm_pretrain(cond, splits, a, tag)
    res = cnn.finetune(splits, a, tag, encoder_state=enc)
    res['msm_val_loss_first'] = hist[0]['msm_val_loss']
    res['msm_val_loss_last'] = hist[-1]['msm_val_loss']
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True, choices=['ast', 'cnn'])
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--conds', nargs='+', default=CONDICOES)
    p.add_argument('--folds', type=int, nargs='+', default=list(range(N_FOLDS)))
    p.add_argument('--teste', action='store_true',
                   help='Ensaio: 1 epoca por estagio, estado/resultados em arquivos _teste.')
    args = p.parse_args()
    global STATE_FILE, RESULTS_FILE
    if args.teste:
        TESTE['on'] = True
        STATE_FILE = STATE_FILE.replace('.json', '_teste.json')
        RESULTS_FILE = RESULTS_FILE.replace('.json', '_teste.json')

    setup_logging()
    state = load_json(STATE_FILE, {})
    results = load_json(RESULTS_FILE, {})
    labels = checar_alinhamento()
    fs = folds(labels)
    for i, (tr, va, te) in enumerate(fs):
        log.info('fold %d: treino %d | validacao %d | teste %d', i, len(tr), len(va), len(te))

    rodar = rodar_ast if args.arch == 'ast' else rodar_cnn
    jobs = [(s, f, c) for s in args.seeds for f in args.folds for c in args.conds]
    log.info('=' * 70)
    log.info('k-fold %s | %d execucoes | sementes %s | folds %s',
             args.arch, len(jobs), args.seeds, args.folds)

    for seed, f, cond in jobs:
        key = '%s|f%d|%s|s%d' % (args.arch, f, cond, seed)
        if state.get(key, {}).get('done'):
            continue
        if free_gb() < MIN_FREE_GB:
            log.error('Disco abaixo de %.1f GB - abortando.', MIN_FREE_GB)
            return 1
        tag = 'kf-%s-f%d-%s-s%d' % (args.arch, f, cond, seed)
        log.info('-' * 70)
        log.info('INICIANDO %s', key)
        t0 = time.time()
        try:
            res = rodar(cond, fs[f], seed, tag)
        except Exception as exc:
            log.exception('[FALHA] %s: %s', key, exc)
            state[key] = {'done': False, 'error': str(exc)}
            save_json(STATE_FILE, state)
            torch.cuda.empty_cache()
            continue
        res.update({'arch': args.arch, 'fold': f, 'condition': cond, 'seed': seed,
                    'test_idx': [int(i) for i in fs[f][2]],
                    'minutes': (time.time() - t0) / 60.0,
                    'when': datetime.datetime.now().isoformat(timespec='seconds')})
        out_dir = 'models_kfold_%s' % args.arch
        preds = os.path.join(out_dir, tag, 'test_preds.json')
        if os.path.exists(preds):
            res['test_preds'] = load_json(preds, None)
        results[key] = res
        state[key] = {'done': True}
        save_json(RESULTS_FILE, results)
        save_json(STATE_FILE, state)
        log.info('CONCLUIDO %s em %.1f min | acc=%.4f f1=%.4f',
                 key, res['minutes'], res['test_accuracy'], res['test_f1_macro'])
    log.info('FIM k-fold %s.', args.arch)
    return 0


if __name__ == '__main__':
    sys.exit(main())
