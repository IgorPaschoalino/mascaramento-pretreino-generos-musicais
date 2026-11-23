"""
run_v2_experiment.py
====================
Orquestrador da reformulacao metodologica v2 (sessao 26/08). Sem argumentos
obrigatorios; resumivel via experiment_state_v2.json.

Escada de ablacao (atribuicao limpa de cada mudanca):

  E0  remontagem bugada   + mascara no fine-tune   -> ja existente (sessoes antigas)
  E1  remontagem CORRETA  + mascara no fine-tune   -> isola o efeito do bug
  E2  remontagem CORRETA  + MSM + fine-tune LIMPO  -> isola o efeito do protocolo

E1 replica exatamente os hiperparametros do E0 do GTZAN (e15, b8, lr5e-5,
mr0.75, alpha0.5, temperature0.1, seed42), mudando SO a remontagem.

E2 = adaptacao fiel a Belfathi et al. (2025):
  Estagio 1 (MSM): mascara pela estrategia, reconstroi o conteudo original dos
    patches mascarados (MSE so nas posicoes mascaradas).
  Estagio 2: classificacao em espectrograma LIMPO, protocolo IDENTICO para as
    4 estrategias. A unica diferenca entre elas e o que foi mascarado no
    Estagio 1 - exatamente o desenho do artigo original.

Uso:
    venv\\Scripts\\python.exe run_v2_experiment.py
    venv\\Scripts\\python.exe run_v2_experiment.py --only e2 --strategies random combined
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
from torch.utils.data import DataLoader
from datasets import DatasetDict
from transformers import (
    ASTForAudioClassification,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup,
)

from masking.patch_mask_collator import cache_strategy_of
from v2_common import (
    AST_CHECKPOINT,
    CleanCollator,
    make_msm_collator,
    ASTForMaskedSpectrogramModeling,
    compute_metrics,
)
from masking.patch_mask_collator import DataCollatorForSelectivePatchMasking
from preprocessing.preprocessing import GTZAN_GENRES

STRATEGIES = ['random', 'spectral_energy', 'onset', 'combined']
STATE_FILE = 'experiment_state_v2.json'
RESULTS_FILE = 'results_v2.json'
LOG_DIR = 'logs_v2'
OUT_DIR = 'models_v2'
MIN_FREE_GB = 3.0

log = logging.getLogger('v2')


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(LOG_DIR, 'run_v2.log'), mode='a', encoding='utf-8')
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
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def cleanup_checkpoints(d):
    """Remove checkpoint-* intermediarios, preservando best_model/."""
    if not os.path.isdir(d):
        return
    for name in os.listdir(d):
        if name.startswith('checkpoint-'):
            shutil.rmtree(os.path.join(d, name), ignore_errors=True)


def device_of():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_ds(dataset, strategy, seed=42):
    if dataset == 'gtzan':
        # inverse_energy le os scores de energia; specaugment usa o cache do
        # random (nao usa score). Os patches sao os mesmos em todos os caches.
        path = 'cache/gtzan-%s-p16-s42' % cache_strategy_of(strategy)
    else:
        raise ValueError('dataset desconhecido: %s' % dataset)
    return DatasetDict.load_from_disk(path).with_format('numpy')


def genres_of(dataset):
    if dataset == 'gtzan':
        return GTZAN_GENRES
    raise ValueError(dataset)


def fresh_classifier(dataset):
    genres = genres_of(dataset)
    return ASTForAudioClassification.from_pretrained(
        AST_CHECKPOINT,
        num_labels=len(genres),
        label2id={g: i for i, g in enumerate(genres)},
        id2label={i: g for i, g in enumerate(genres)},
        ignore_mismatched_sizes=True,
    )


# ---------------------------------------------------------------------------
# Estagio 1: pre-treino MSM (loop manual - loss customizada)
# ---------------------------------------------------------------------------

def msm_pretrain(strategy, ds, args, tag):
    """Treina o encoder AST a reconstruir os patches mascarados.

    Devolve o caminho do state_dict do encoder adaptado.
    """
    dev = device_of()
    grid = tuple(int(x) for x in np.asarray(ds['train'][0]['grid_shape']))

    model = ASTForMaskedSpectrogramModeling(grid_shape=grid).to(dev)

    train_col = make_msm_collator(strategy, args.mask_ratio, args.mask_choice,
                                  args.temperature, args.seed)
    train_dl = DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True,
                          collate_fn=train_col, num_workers=0, drop_last=False)

    opt = torch.optim.AdamW([
        {'params': model.ast.parameters(), 'lr': args.msm_lr},
        {'params': model.decoder.parameters(), 'lr': args.msm_lr * 10},
    ], weight_decay=0.01)

    total = args.msm_epochs * len(train_dl)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))

    history = []
    for ep in range(1, args.msm_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for batch in train_dl:
            iv = batch['input_values'].to(dev, non_blocking=True)
            cp = batch['clean_patches'].to(dev, non_blocking=True)
            mk = batch['mask'].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
                loss = model(iv, cp, mk)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.item()) * iv.shape[0]
            n += iv.shape[0]
        train_loss = tot / max(n, 1)

        # validacao com mascara de semente fixa (comparavel entre epocas)
        val_col = make_msm_collator(strategy, args.mask_ratio, args.mask_choice,
                                    args.temperature, args.seed + 1000)
        val_dl = DataLoader(ds['validation'], batch_size=args.batch_size,
                            shuffle=False, collate_fn=val_col, num_workers=0)
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                iv = batch['input_values'].to(dev)
                cp = batch['clean_patches'].to(dev)
                mk = batch['mask'].to(dev)
                with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
                    loss = model(iv, cp, mk)
                tot += float(loss.item()) * iv.shape[0]
                n += iv.shape[0]
        val_loss = tot / max(n, 1)

        history.append({'epoch': ep, 'msm_train_loss': train_loss, 'msm_val_loss': val_loss})
        log.info('[MSM %s] epoca %2d/%d  train=%.4f  val=%.4f',
                 tag, ep, args.msm_epochs, train_loss, val_loss)

    enc_path = os.path.join(OUT_DIR, tag, 'msm_encoder.pt')
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)
    torch.save(model.ast.state_dict(), enc_path)
    save_json(os.path.join(OUT_DIR, tag, 'msm_history.json'), history)

    del model
    torch.cuda.empty_cache()
    return enc_path, history


# ---------------------------------------------------------------------------
# Estagio 2 / E1: fine-tuning de classificacao
# ---------------------------------------------------------------------------

def finetune(ds, args, tag, collator, eval_collator, dataset,
             encoder_state=None):
    """Fine-tuning de classificacao. `collator` e usado no treino e na
    validacao; `eval_collator` na avaliacao final de teste."""
    out = os.path.join(OUT_DIR, tag)
    os.makedirs(out, exist_ok=True)

    model = fresh_classifier(dataset)
    if encoder_state is not None:
        sd = torch.load(encoder_state, map_location='cpu')
        missing, unexpected = model.audio_spectrogram_transformer.load_state_dict(
            sd, strict=False)
        log.info('[%s] encoder MSM carregado (missing=%d unexpected=%d)',
                 tag, len(missing), len(unexpected))
        if len(missing) or len(unexpected):
            log.warning('[%s] missing=%s unexpected=%s', tag,
                        list(missing)[:5], list(unexpected)[:5])
    model.to(device_of())

    targs = TrainingArguments(
        output_dir=out,
        num_train_epochs=args.ft_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.ft_lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        save_total_limit=1,
        logging_dir=os.path.join(out, 'runs'),
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        report_to='none',
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    curve = [
        {'epoch': r.get('epoch'), 'eval_loss': r.get('eval_loss'),
         'eval_f1_macro': r.get('eval_f1_macro'), 'eval_accuracy': r.get('eval_accuracy')}
        for r in trainer.state.log_history if 'eval_loss' in r
    ]

    trainer.data_collator = eval_collator
    pred = trainer.predict(ds['test'], metric_key_prefix='test')
    test = pred.metrics

    # Predicoes por exemplo: sem elas nao ha teste estatistico pareado
    # (McNemar) entre estrategias. Salvar sempre - os classificadores
    # fine-tunados sao descartados, entao isto e a unica chance.
    save_json(os.path.join(out, 'test_preds.json'), {
        'y_true': [int(x) for x in np.asarray(pred.label_ids).ravel()],
        'y_pred': [int(x) for x in np.asarray(pred.predictions).argmax(-1)],
    })

    res = {
        'test_accuracy': float(test.get('test_accuracy', float('nan'))),
        'test_f1_macro': float(test.get('test_f1_macro', float('nan'))),
        'test_loss': float(test.get('test_loss', float('nan'))),
        'curve': curve,
    }
    save_json(os.path.join(out, 'result.json'), res)

    del model, trainer
    torch.cuda.empty_cache()
    cleanup_checkpoints(out)
    return res


# ---------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------

def run_e1(strategy, args):
    """Remontagem CORRIGIDA, resto identico ao E0 (mascara tambem no teste)."""
    tag = 'e1-%s-%s%s' % (args.dataset, strategy, args.tag_suffix)
    ds = load_ds(args.dataset, strategy, args.seed)

    kw = dict(mask_ratio=args.mask_ratio, seed=args.seed, correct_reassembly=True)
    if strategy == 'random':
        col = DataCollatorForSelectivePatchMasking(
            mask_choice_strategy='weighted_random', temperature=1e6, **kw)
    else:
        col = DataCollatorForSelectivePatchMasking(
            mask_choice_strategy=args.mask_choice, temperature=args.temperature, **kw)

    log.info('[E1 %s] fine-tuning com mascara (remontagem corrigida)', strategy)
    res = finetune(ds, args, tag, collator=col, eval_collator=col,
                   dataset=args.dataset)

    # avaliacao limpa adicional, para comparar com o E2 na mesma escala
    log.info('[E1 %s] test(mascarado) acc=%.4f f1=%.4f',
             strategy, res['test_accuracy'], res['test_f1_macro'])
    return res


def run_e2(strategy, args):
    """MSM (estagio 1) + fine-tuning LIMPO (estagio 2) + avaliacao limpa."""
    tag = 'e2-%s-%s%s' % (args.dataset, strategy, args.tag_suffix)
    ds = load_ds(args.dataset, strategy, args.seed)

    log.info('[E2 %s] Estagio 1: pre-treino MSM (%d epocas, mr=%.2f)',
             strategy, args.msm_epochs, args.mask_ratio)
    enc, hist = msm_pretrain(strategy, ds, args, tag)
    log.info('[E2 %s] Estagio 1 concluido. val_loss inicial=%.4f final=%.4f',
             strategy, hist[0]['msm_val_loss'], hist[-1]['msm_val_loss'])

    clean = CleanCollator()
    log.info('[E2 %s] Estagio 2: fine-tuning LIMPO (%d epocas)', strategy, args.ft_epochs)
    res = finetune(ds, args, tag, collator=clean, eval_collator=clean,
                   dataset=args.dataset, encoder_state=enc)
    res['msm_val_loss_first'] = hist[0]['msm_val_loss']
    res['msm_val_loss_last'] = hist[-1]['msm_val_loss']
    log.info('[E2 %s] test(limpo) acc=%.4f f1=%.4f',
             strategy, res['test_accuracy'], res['test_f1_macro'])
    return res


def run_e2_noadapt(strategy, args):
    """Controle do E2: fine-tuning LIMPO sem nenhum pre-treino MSM.

    Serve para responder 'o MSM ajudou alguma coisa?' - e o piso comum das 4
    estrategias, ja que sem MSM a estrategia de mascaramento e irrelevante.
    """
    tag = 'e2ctl-%s-noadapt%s' % (args.dataset, args.tag_suffix)
    ds = load_ds(args.dataset, 'random', args.seed)
    clean = CleanCollator()
    log.info('[E2-ctl] fine-tuning LIMPO sem MSM (controle)')
    res = finetune(ds, args, tag, collator=clean, eval_collator=clean,
                   dataset=args.dataset)
    log.info('[E2-ctl] test(limpo) acc=%.4f f1=%.4f',
             res['test_accuracy'], res['test_f1_macro'])
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='gtzan')
    p.add_argument('--strategies', nargs='+', default=STRATEGIES)
    p.add_argument('--only', default='all', choices=['all', 'e1', 'e2', 'ctl'])
    p.add_argument('--mask-ratio', type=float, default=0.75)
    p.add_argument('--mask-choice', default='weighted_random')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--ft-epochs', type=int, default=15)
    p.add_argument('--ft-lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--msm-epochs', type=int, default=30)
    p.add_argument('--msm-lr', type=float, default=5e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='Varias sementes numa UNICA invocacao python: um so processo faz tudo, sem spawnar nada. Um python orfao neste harness perde a console e nao consegue criar subprocessos (STATUS_DLL_INIT_FAILED). O sufixo de tag vira <tag_suffix>-s<seed> por semente.')
    p.add_argument('--tag-suffix', default='')
    return p.parse_args()


def main():
    args = arguments()
    setup_logging()
    os.makedirs(OUT_DIR, exist_ok=True)

    state = load_json(STATE_FILE, {})
    results = load_json(RESULTS_FILE, {})

    log.info('=' * 70)
    log.info('run_v2_experiment | dataset=%s only=%s strategies=%s',
             args.dataset, args.only, args.strategies)
    log.info('mask_ratio=%.2f temperature=%.3f msm_epochs=%d ft_epochs=%d batch=%d',
             args.mask_ratio, args.temperature, args.msm_epochs,
             args.ft_epochs, args.batch_size)
    log.info('disco livre: %.1f GB', free_gb())
    log.info('=' * 70)

    jobs = []
    if args.only in ('all', 'ctl'):
        jobs.append(('ctl', 'noadapt'))
    if args.only in ('all', 'e1'):
        jobs += [('e1', s) for s in args.strategies]
    if args.only in ('all', 'e2'):
        jobs += [('e2', s) for s in args.strategies]

    seeds = args.seeds if args.seeds else [args.seed]
    base_suffix = args.tag_suffix
    multi = len(seeds) > 1

    for seed in seeds:
        args.seed = seed
        args.tag_suffix = base_suffix + ('-s%d' % seed if multi else '')
        if multi:
            log.info('#' * 70)
            log.info('SEMENTE %d (sufixo %s)', seed, args.tag_suffix)
            log.info('#' * 70)
        run_jobs(jobs, args, state, results)

    log.info('=' * 70)
    log.info('Todos os jobs processados. Resultados em %s', RESULTS_FILE)


def run_jobs(jobs, args, state, results):
    for kind, strat in jobs:
        key = '%s|%s|%s%s' % (kind, args.dataset, strat, args.tag_suffix)
        if state.get(key, {}).get('done'):
            log.info('[skip] %s ja concluido', key)
            continue
        if free_gb() < MIN_FREE_GB:
            log.error('Disco abaixo de %.1f GB - abortando.', MIN_FREE_GB)
            break

        t0 = time.time()
        log.info('-' * 70)
        log.info('INICIANDO %s', key)
        try:
            if kind == 'e1':
                res = run_e1(strat, args)
            elif kind == 'e2':
                res = run_e2(strat, args)
            else:
                res = run_e2_noadapt(strat, args)
        except Exception as exc:
            log.exception('[FALHA] %s: %s', key, exc)
            state[key] = {'done': False, 'error': str(exc)}
            save_json(STATE_FILE, state)
            torch.cuda.empty_cache()
            continue

        dt = (time.time() - t0) / 60.0
        res['minutes'] = dt
        res['when'] = datetime.datetime.now().isoformat(timespec='seconds')
        results[key] = res
        state[key] = {'done': True}
        save_json(RESULTS_FILE, results)
        save_json(STATE_FILE, state)
        log.info('CONCLUIDO %s em %.1f min | acc=%.4f f1=%.4f',
                 key, dt, res['test_accuracy'], res['test_f1_macro'])
        log.info('disco livre: %.1f GB', free_gb())


if __name__ == '__main__':
    main()
