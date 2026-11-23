"""
run_cnn_experiment.py
=====================
Replicacao de arquitetura: o mesmo protocolo E2 (MSM + fine-tuning limpo),
trocando o AST por uma CNN treinada do zero. Orquestrador resumivel, sem
argumentos obrigatorios.

A pergunta
----------
No AST, as quatro estrategias de mascaramento colapsaram na mesma acuracia, e
no mesmo valor do controle sem MSM. Ha duas leituras possiveis:

  (a) o mascaramento seletivo nao ajuda - resultado sobre o METODO;
  (b) o AST, ja pre-treinado em ~2M clipes do AudioSet, nao se move com 20
      epocas de MSM sobre 699 clipes do GTZAN - resultado sobre o PONTO DE
      PARTIDA, que nada diz sobre o metodo.

Uma CNN sem nenhum pre-treino separa as duas: aqui o MSM e a unica fonte de
representacao antes do fine-tuning. Se as estrategias continuarem empatadas, a
leitura (a) se fortalece; se separarem, o resultado nulo do AST era saturacao.

O desenho tambem corrige duas fraquezas estatisticas dos experimentos com AST:

  - **poder**: a CNN tem ~1,2M parametros contra ~86M do AST, entao cada
    execucao custa minutos em vez de ~45 min. Isso viabiliza 5 sementes por
    condicao em vez de 1-3, que e o que falta para qualquer teste ter poder.
  - **teste pareado**: as predicoes por exemplo sao salvas (`test_preds.json`),
    o que permite McNemar entre estrategias na MESMA semente - bem mais
    sensivel que comparar acuracias agregadas.

Condicoes: {random, spectral_energy, onset, combined} + controle sem MSM,
cada uma em N sementes.

Uso:
    venv\\Scripts\\python.exe run_cnn_experiment.py
    venv\\Scripts\\python.exe run_cnn_experiment.py --seeds 42 43 --smoke
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
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_cosine_schedule_with_warmup

from cnn_common import CNNForMaskedSpectrogramModeling, CNNClassifier
from cnn_data import (
    load_splits, iter_batches, MaskSampler, msm_batch, clean_batch,
)
from preprocessing.preprocessing import GTZAN_GENRES

STRATEGIES = ['random', 'spectral_energy', 'onset', 'combined']
STATE_FILE = 'experiment_state_cnn.json'
RESULTS_FILE = 'results_cnn.json'
LOG_DIR = 'logs_cnn'
OUT_DIR = 'models_cnn'
MIN_FREE_GB = 3.0

log = logging.getLogger('cnn')


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(LOG_DIR, 'run_cnn.log'), mode='a',
                             encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def free_gb(path='.'):
    return shutil.disk_usage(os.path.abspath(path)).free / (1024 ** 3)


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def device_of():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)





# ---------------------------------------------------------------------------
# Estagio 1: pre-treino MSM
# ---------------------------------------------------------------------------

def msm_pretrain(strategy, splits, args, tag):
    """Treina o encoder da CNN a reconstruir os patches mascarados."""
    dev = device_of()
    model = CNNForMaskedSpectrogramModeling().to(dev)

    tr, va = splits['train'], splits['validation']
    uniform = (strategy == 'random')
    train_sampler = MaskSampler(args.mask_ratio, args.mask_choice,
                                args.temperature, args.seed, uniform,
                                strategy=strategy, grid=tr.grid)
    opt = torch.optim.AdamW(model.parameters(), lr=args.msm_lr, weight_decay=0.01)
    steps_per_epoch = max(int(np.ceil(len(tr) / args.batch_size)), 1)
    total = args.msm_epochs * steps_per_epoch
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))
    rng = np.random.default_rng(args.seed)

    history = []
    for ep in range(1, args.msm_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for idx in iter_batches(len(tr), args.batch_size, rng):
            x, cp, mk = msm_batch(tr, idx, train_sampler, dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
                loss = model(x, cp, mk)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        train_loss = tot / max(n, 1)

        # mascara de validacao com semente fixa por epoca: as perdas de
        # validacao ficam comparaveis entre epocas e entre execucoes.
        val_sampler = MaskSampler(args.mask_ratio, args.mask_choice,
                                  args.temperature, args.seed + 1000, uniform,
                                  strategy=strategy, grid=va.grid)
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for idx in iter_batches(len(va), args.batch_size):
                x, cp, mk = msm_batch(va, idx, val_sampler, dev)
                with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
                    loss = model(x, cp, mk)
                tot += float(loss.item()) * x.shape[0]
                n += x.shape[0]
        val_loss = tot / max(n, 1)

        history.append({'epoch': ep, 'msm_train_loss': train_loss,
                        'msm_val_loss': val_loss})
        if ep == 1 or ep % 5 == 0 or ep == args.msm_epochs:
            log.info('[MSM %s] epoca %2d/%d  train=%.5f  val=%.5f',
                     tag, ep, args.msm_epochs, train_loss, val_loss)

    state = {k: v.cpu() for k, v in model.encoder.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    return state, history


# ---------------------------------------------------------------------------
# Estagio 2: fine-tuning de classificacao
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, split, batch_size, dev):
    model.eval()
    logits, labels = [], []
    for idx in iter_batches(len(split), batch_size):
        x, y = clean_batch(split, idx, dev)
        with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
            out = model(x)
        logits.append(out.float().cpu())
        labels.append(y.cpu())
    logits = torch.cat(logits).numpy()
    labels = torch.cat(labels).numpy()
    preds = logits.argmax(-1)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1_macro': float(f1_score(labels, preds, average='macro', zero_division=0)),
    }, preds, labels


def finetune(splits, args, tag, encoder_state=None):
    """Fine-tuning limpo. Protocolo IDENTICO para todas as condicoes - a unica
    diferenca entre elas e o encoder com que se comeca."""
    dev = device_of()
    out_dir = os.path.join(OUT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    model = CNNClassifier(num_labels=len(GTZAN_GENRES)).to(dev)
    if encoder_state is not None:
        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        log.info('[%s] encoder MSM carregado (missing=%d unexpected=%d)',
                 tag, len(missing), len(unexpected))

    tr, va, te = splits['train'], splits['validation'], splits['test']
    opt = torch.optim.AdamW(model.parameters(), lr=args.ft_lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = max(int(np.ceil(len(tr) / args.batch_size)), 1)
    total = args.ft_epochs * steps_per_epoch
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    rng = np.random.default_rng(args.seed + 500)

    best_f1, best_state, curve = -1.0, None, []
    for ep in range(1, args.ft_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for idx in iter_batches(len(tr), args.batch_size, rng):
            x, y = clean_batch(tr, idx, dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.item()) * x.shape[0]
            n += x.shape[0]

        m, _, _ = evaluate(model, va, args.batch_size, dev)
        curve.append({'epoch': ep, 'train_loss': tot / max(n, 1),
                      'eval_accuracy': m['accuracy'], 'eval_f1_macro': m['f1_macro']})
        # selecao de modelo pela validacao, igual ao protocolo do AST
        # (load_best_model_at_end + metric_for_best_model='f1_macro')
        if m['f1_macro'] > best_f1:
            best_f1 = m['f1_macro']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 5 == 0 or ep == args.ft_epochs:
            log.info('[FT %s] epoca %2d/%d  loss=%.4f  val_acc=%.4f  val_f1=%.4f',
                     tag, ep, args.ft_epochs, tot / max(n, 1), m['accuracy'], m['f1_macro'])

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m, preds, labels = evaluate(model, te, args.batch_size, dev)

    # Predicoes por exemplo: sem elas nao ha McNemar pareado entre estrategias.
    save_json(os.path.join(out_dir, 'test_preds.json'),
              {'y_true': [int(v) for v in labels], 'y_pred': [int(v) for v in preds]})

    res = {'test_accuracy': test_m['accuracy'], 'test_f1_macro': test_m['f1_macro'],
           'best_val_f1': best_f1, 'curve': curve}
    save_json(os.path.join(out_dir, 'result.json'), res)

    del model
    torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------
# Condicoes
# ---------------------------------------------------------------------------

def run_msm_condition(strategy, args, tag):
    splits = load_splits(strategy, args.max_examples)
    log.info('[%s] Estagio 1: MSM (%d epocas, mr=%.2f)', tag, args.msm_epochs, args.mask_ratio)
    enc, hist = msm_pretrain(strategy, splits, args, tag)
    log.info('[%s] Estagio 1 ok. val_loss %.5f -> %.5f',
             tag, hist[0]['msm_val_loss'], hist[-1]['msm_val_loss'])
    save_json(os.path.join(OUT_DIR, tag, 'msm_history.json'), hist)
    log.info('[%s] Estagio 2: fine-tuning LIMPO (%d epocas)', tag, args.ft_epochs)
    res = finetune(splits, args, tag, encoder_state=enc)
    res['msm_val_loss_first'] = hist[0]['msm_val_loss']
    res['msm_val_loss_last'] = hist[-1]['msm_val_loss']
    return res


def run_control(args, tag):
    """Sem MSM: encoder aleatorio direto para o fine-tuning. E o piso comum -
    responde 'o MSM ajudou alguma coisa?' antes de comparar estrategias."""
    splits = load_splits('random', args.max_examples)
    log.info('[%s] controle: fine-tuning LIMPO sem MSM', tag)
    return finetune(splits, args, tag, encoder_state=None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--strategies', nargs='+', default=STRATEGIES)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                   help='Varias sementes numa UNICA invocacao python (um so '
                        'processo, sem spawnar subprocessos - ver PROGRESS.md, '
                        'INCIDENTE 3 de 26/08).')
    p.add_argument('--no-control', action='store_true')
    p.add_argument('--mask-ratio', type=float, default=0.75)
    p.add_argument('--mask-choice', default='weighted_random')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--msm-epochs', type=int, default=30)
    p.add_argument('--msm-lr', type=float, default=1e-3)
    p.add_argument('--ft-epochs', type=int, default=40)
    p.add_argument('--ft-lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--max-examples', type=int, default=None)
    p.add_argument('--tag-suffix', default='')
    p.add_argument('--smoke', action='store_true',
                   help='1 semente, poucas epocas e poucos exemplos - so valida shapes.')
    return p.parse_args()


def main():
    args = arguments()
    if args.smoke:
        # Dataset COMPLETO, poucas epocas: alem de validar shapes, mede o
        # tempo real por epoca. Nao usar `--max-examples` aqui - o cache esta
        # ordenado por classe, entao um recorte pequeno pega uma classe so e
        # a acuracia sai zerada por construcao, mascarando problemas reais.
        args.seeds = [42]
        args.msm_epochs = 2
        args.ft_epochs = 2
        args.strategies = args.strategies[:1]
        args.tag_suffix = args.tag_suffix + '-smoke'

    setup_logging()
    os.makedirs(OUT_DIR, exist_ok=True)
    state = load_json(STATE_FILE, {})
    results = load_json(RESULTS_FILE, {})

    log.info('=' * 70)
    log.info('run_cnn_experiment | estrategias=%s sementes=%s smoke=%s',
             args.strategies, args.seeds, args.smoke)
    log.info('mr=%.2f msm_epochs=%d ft_epochs=%d batch=%d | disco %.1f GB | dev=%s',
             args.mask_ratio, args.msm_epochs, args.ft_epochs, args.batch_size,
             free_gb(), device_of())
    log.info('=' * 70)

    conditions = ([] if args.no_control else ['noadapt']) + list(args.strategies)

    for seed in args.seeds:
        log.info('#' * 70)
        log.info('SEMENTE %d', seed)
        log.info('#' * 70)
        for cond in conditions:
            tag = 'cnn-gtzan-%s%s-s%d' % (cond, args.tag_suffix, seed)
            key = 'cnn|gtzan|%s%s-s%d' % (cond, args.tag_suffix, seed)
            if state.get(key, {}).get('done'):
                log.info('[skip] %s ja concluido', key)
                continue
            if free_gb() < MIN_FREE_GB:
                log.error('Disco abaixo de %.1f GB - abortando.', MIN_FREE_GB)
                return

            args.seed = seed
            set_seed(seed)
            t0 = time.time()
            log.info('-' * 70)
            log.info('INICIANDO %s', key)
            try:
                if cond == 'noadapt':
                    res = run_control(args, tag)
                else:
                    res = run_msm_condition(cond, args, tag)
            except Exception as exc:
                log.exception('[FALHA] %s: %s', key, exc)
                state[key] = {'done': False, 'error': str(exc)}
                save_json(STATE_FILE, state)
                torch.cuda.empty_cache()
                continue

            res['minutes'] = (time.time() - t0) / 60.0
            res['when'] = datetime.datetime.now().isoformat(timespec='seconds')
            res['seed'] = seed
            res['condition'] = cond
            results[key] = res
            state[key] = {'done': True}
            save_json(RESULTS_FILE, results)
            save_json(STATE_FILE, state)
            log.info('CONCLUIDO %s em %.1f min | acc=%.4f f1=%.4f',
                     key, res['minutes'], res['test_accuracy'], res['test_f1_macro'])

    log.info('=' * 70)
    log.info('FIM. Resultados em %s', RESULTS_FILE)


if __name__ == '__main__':
    main()
