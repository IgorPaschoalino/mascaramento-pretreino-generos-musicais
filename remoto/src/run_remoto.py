"""
run_remoto.py
=============
Orquestrador da execucao remota. Roda o protocolo de DOIS ESTAGIOS validado no
GTZAN sobre FMA-small e Homburg, com as 6 estrategias (4 originais + 2 de
referencia: inverse_energy e specaugment) mais o controle.

v2 (15/10): no Homburg, se o conjunto de teste for o mesmo da corrida v1, roda
so as 2 estrategias novas (o resto e reaproveitado). No FMA, bloco completo.

Uso normal (nao precisa de argumento nenhum):
    python run_remoto.py

Teste de fumaca (obrigatorio antes da corrida real -- ~15 min, prova o pipeline
de ponta a ponta em escala minima):
    python run_remoto.py --fumaca

Ver o plano e o tempo estimado sem rodar nada:
    python run_remoto.py --plano

DESENHO, e por que ele e assim:

  * Ordem por BLOCO COMPLETO DE SEMENTE, nao por condicao. Os testes usados na
    analise (Friedman, t pareado por semente) sao testes por blocos: uma
    semente a mais COMPLETA (controle + 4 estrategias) e imediatamente
    utilizavel, enquanto uma estrategia a mais espalhada por varias sementes
    nao e. Se a corrida for interrompida no meio, o que ja rodou tem valor
    estatistico. Esta licao custou uma corrida perdida ao projeto.

  * Homburg ANTES do FMA-small. O Homburg custa ~1/5 do FMA e fecha sozinho a
    lacuna de "generalizacao para outra taxonomia". O FMA e o resultado mais
    valioso (melhor poder estatistico do projeto) mas so e citavel com 6
    sementes; deixa-lo por ultimo garante que uma interrupcao nao deixe o
    usuario sem nenhum resultado completo.

  * Totalmente resumivel. `estado_remoto.json` marca cada execucao concluida;
    reinvocar o script pula o que ja terminou. Seguro rodar as cegas.

  * O encoder MSM (~340 MB) e APAGADO depois do Estagio 2. Sao 5 condicoes x 6
    sementes x 2 datasets = ate 60 arquivos; manter todos custaria ~20 GB sem
    nenhum uso posterior (o que importa e `result.json` + `test_preds.json`).

  * Hiperparametros TRAVADOS nos do GTZAN (batch 8, mask_ratio 0.75,
    temperature 0.1, alpha 0.5, 30 epocas MSM, 15 de fine-tuning, lr 5e-5).
    Em particular NAO aumentar o batch para aproveitar a VRAM da placa: o
    batch faz parte do protocolo, e mudar confundiria a comparacao com o GTZAN.
"""

import os
import sys
import json
import time
import shutil
import logging
import pickle
import argparse
import datetime
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import (
    TrainingArguments, Trainer, TrainerCallback, get_cosine_schedule_with_warmup,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from remoto_common import (
    STRATEGIES, ESTRATEGIAS_NOVAS, DATASETS, CleanCollator, MSMCollator,
    ASTForMaskedSpectrogramModeling, fresh_classifier, compute_metrics,
    cache_path_of, genres_of,
)

RESULTS_DIR = 'resultados'
STATE_FILE = os.path.join(RESULTS_DIR, 'estado_remoto.json')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'resultados_remoto.json')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
RUNS_DIR = os.path.join(RESULTS_DIR, 'execucoes')
WORK_DIR = 'trabalho_tmp'          # checkpoints e encoders, descartaveis
MIN_FREE_GB = 8.0

# Ordem dos datasets: o barato e completavel primeiro (ver docstring).
PLANO_DATASETS = ['homburg', 'fma']
SEEDS_PADRAO = [42, 43, 44, 45, 46, 47]   # mesmas 6 sementes do AST no GTZAN
# Gravado por localizar_dados.py: se o cache do Homburg desta maquina tem o
# MESMO conjunto de teste da corrida v1 (01/10 e 05/10), as 5 condicoes ja
# rodadas la sao reaproveitadas e aqui so entram as 2 estrategias novas.
COMPAT_FILE = os.path.join(RESULTS_DIR, 'compatibilidade.json')

log = logging.getLogger('remoto')


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def setup_logging(nome='run_remoto.log'):
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(LOG_DIR, nome), mode='a', encoding='utf-8')
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
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)          # gravacao atomica: nunca deixa json truncado


def device_of():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Progresso visivel
#
# A corrida leva horas ou dias, e quem esta operando a maquina nao conhece o
# experimento. Sem indicacao de progresso e impossivel distinguir "andando
# normalmente" de "travado", entao o progresso e reportado em tres niveis:
# geral (a corrida toda), por execucao, e por epoca dentro da execucao.
# Tudo em ASCII -- o console do Windows quebra com acentos e emoji.
# ---------------------------------------------------------------------------

PROG = {'total': 0, 'feitos': 0, 'inicio': None, 'rotulo': ''}


def barra(fracao, largura=30):
    fracao = min(max(float(fracao), 0.0), 1.0)
    cheio = int(round(fracao * largura))
    return '[%s%s] %3.0f%%' % ('#' * cheio, '.' * (largura - cheio), 100 * fracao)


def hm(minutos):
    """Minutos -> '3h 05min' ou '42min'."""
    if minutos is None or minutos != minutos or minutos < 0:
        return '?'
    if minutos < 60:
        return '%.0fmin' % minutos
    return '%dh %02dmin' % (int(minutos // 60), int(round(minutos % 60)))


def log_geral():
    """Linha de progresso da corrida inteira."""
    if not PROG['total'] or PROG['inicio'] is None:
        return
    feitos, total = PROG['feitos'], PROG['total']
    decorrido = (time.time() - PROG['inicio']) / 60.0
    if feitos:
        resta = decorrido / feitos * (total - feitos)
        previsao = (datetime.datetime.now()
                    + datetime.timedelta(minutes=resta)).strftime('%d/%m %H:%M')
    else:
        resta, previsao = float('nan'), '?'
    log.info('GERAL    %s  %d/%d execucoes | decorrido %s | resta ~%s | termina ~%s',
             barra(feitos / total), feitos, total, hm(decorrido), hm(resta), previsao)


def log_etapa(rotulo, feito, total, extra=''):
    """Linha de progresso dentro de uma execucao (epocas de MSM ou fine-tuning)."""
    log.info('  %-9s %s  %d/%d%s', rotulo, barra(feito / max(total, 1)),
             feito, total, ('  ' + extra) if extra else '')


def ambiente():
    """Impressao digital do ambiente -- vai junto com os resultados para que a
    analise saiba em que maquina cada numero foi produzido."""
    info = {
        'quando': datetime.datetime.now().isoformat(timespec='seconds'),
        'python': sys.version.split()[0],
        'plataforma': platform.platform(),
        'torch': torch.__version__,
        'cuda_disponivel': torch.cuda.is_available(),
    }
    try:
        import transformers, datasets as _ds, numpy as _np
        info.update({'transformers': transformers.__version__,
                     'datasets': _ds.__version__, 'numpy': _np.__version__})
    except Exception:
        pass
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['vram_gb'] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
        info['cuda'] = torch.version.cuda
    return info


def limpar_checkpoints(d):
    if not os.path.isdir(d):
        return
    for name in os.listdir(d):
        if name.startswith('checkpoint-'):
            shutil.rmtree(os.path.join(d, name), ignore_errors=True)


def carregar_dataset(dataset, max_examples=None):
    """Carrega o cache do dataset, opcionalmente subamostrado.

    `max_examples` subamostra o cache COMPLETO na carga, em vez de procurar um
    cache pequeno separado. E o que faz o teste de fumaca exercitar exatamente
    os mesmos dados da corrida real -- se houvesse um cache proprio para o
    ensaio, o ensaio poderia passar e a corrida falhar por um problema que so
    existe nos dados de verdade.
    """
    base = cache_path_of(dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            'Cache nao encontrado: %s\n'
            'Rode primeiro:  python preprocess_remoto.py --dataset %s'
            % (base, dataset))
    d = {}
    for s in ('train', 'validation', 'test'):
        ds = Dataset.load_from_disk(os.path.join(base, s))
        if max_examples and len(ds) > max_examples:
            # Embaralha antes de cortar: os splits estao agrupados por classe em
            # parte, e pegar os N primeiros poderia deixar generos de fora --
            # um ensaio com 3 das 9 classes nao provaria muita coisa.
            ds = ds.shuffle(seed=0).select(range(max_examples))
        d[s] = ds
    return DatasetDict(d).with_format('numpy')


# ---------------------------------------------------------------------------
# Pausa e retomada
#
# A corrida pode ser interrompida a qualquer momento e retomada com o MESMO
# comando. O progresso e salvo ao fim de cada EPOCA (nao so de cada execucao):
# no FMA uma execucao leva horas, mas uma epoca leva minutos, entao interromper
# perde no maximo uma epoca.
#
# Tres formas de parar:
#   - Ctrl+C (ou fechar o terminal): para na hora; perde a epoca em andamento.
#   - criar o arquivo PAUSAR nesta pasta (comando: touch PAUSAR): termina a
#     epoca em andamento, salva e sai limpo.
#   - --parar-as HH:MM: o mesmo, automaticamente, ao passar desse horario
#     (para rodar so durante a noite, por exemplo).
# ---------------------------------------------------------------------------

ARQUIVO_PAUSA = 'PAUSAR'


class PausaSolicitada(Exception):
    """Levantada ao fim de uma epoca ja salva, quando pediram para parar."""


PARAR_EM = {'quando': None}


def configurar_parada(hhmm):
    """--parar-as HH:MM -> proxima ocorrencia desse horario a partir de agora."""
    if not hhmm:
        return
    h, m = (int(x) for x in hhmm.split(':'))
    agora = datetime.datetime.now()
    alvo = agora.replace(hour=h, minute=m, second=0, microsecond=0)
    if alvo <= agora:
        alvo += datetime.timedelta(days=1)
    PARAR_EM['quando'] = alvo
    log.info('Parada automatica programada para %s.', alvo.strftime('%d/%m %H:%M'))


def deve_parar():
    if os.path.exists(ARQUIVO_PAUSA):
        return 'arquivo PAUSAR encontrado'
    if PARAR_EM['quando'] and datetime.datetime.now() >= PARAR_EM['quando']:
        return 'horario de parada (%s) atingido' % PARAR_EM['quando'].strftime('%H:%M')
    return None


def salvar_torch(obj, path):
    """Gravacao atomica: uma interrupcao no meio nunca deixa arquivo corrompido."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Estagio 1: pre-treino MSM
# ---------------------------------------------------------------------------

def msm_pretrain(strategy, ds, args, tag):
    """Treina o encoder AST a reconstruir os patches mascarados.

    Loop manual (nao Trainer) porque a loss e customizada: MSE apenas nas
    posicoes mascaradas.

    Resumivel por epoca: ao fim de cada uma grava em trabalho_tmp/<tag>/ o
    modelo, o otimizador, o scheduler, o scaler, o estado da RNG das mascaras e
    o historico. A ordem dos lotes de cada epoca vem de um gerador semeado por
    (semente, epoca), entao a epoca retomada ve exatamente os mesmos lotes que
    veria sem a interrupcao.
    """
    dev = device_of()
    work = os.path.join(WORK_DIR, tag)
    enc_path = os.path.join(work, 'msm_encoder.pt')
    hist_path = os.path.join(work, 'msm_history.json')
    ckpt_path = os.path.join(work, 'msm_ckpt.pt')

    # Estagio 1 ja concluido numa sessao anterior (interrompida no Estagio 2).
    if os.path.exists(enc_path) and os.path.exists(hist_path):
        log.info('[%s] Estagio 1 ja concluido antes -- reaproveitando o encoder.', tag)
        return enc_path, load_json(hist_path, [])

    grid = tuple(int(x) for x in np.asarray(ds['train'][0]['grid_shape']))
    model = ASTForMaskedSpectrogramModeling(grid_shape=grid).to(dev)

    train_col = MSMCollator(strategy, mask_ratio=args.mask_ratio,
                            mask_choice=args.mask_choice,
                            temperature=args.temperature, alpha=args.alpha,
                            seed=args.seed)
    gerador = torch.Generator()
    train_dl = DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True,
                          collate_fn=train_col, num_workers=0, drop_last=False,
                          generator=gerador)

    opt = torch.optim.AdamW([
        {'params': model.ast.parameters(), 'lr': args.msm_lr},
        {'params': model.decoder.parameters(), 'lr': args.msm_lr * 10},
    ], weight_decay=0.01)

    total = args.msm_epochs * len(train_dl)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))

    history = []
    ep_inicial = 1
    if os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched'])
            scaler.load_state_dict(ck['scaler'])
            train_col.rng.bit_generator.state = ck['rng_mascara']
            history = ck['history']
            ep_inicial = ck['epoca'] + 1
            log.info('[%s] RETOMANDO o Estagio 1 da epoca %d/%d.',
                     tag, ep_inicial, args.msm_epochs)
        except Exception as exc:
            log.warning('[%s] checkpoint do Estagio 1 ilegivel (%s) -- recomecando o estagio.',
                        tag, exc)
            del model, opt
            return _msm_sem_checkpoint(strategy, ds, args, tag, ckpt_path)

    # Mascara de validacao com semente FIXA: as perdas de validacao precisam ser
    # comparaveis entre epocas, entao a mascara nao pode variar. Recriada a cada
    # epoca, para que a retomada nao altere a mascara.
    def val_col():
        return MSMCollator(strategy, mask_ratio=args.mask_ratio,
                           mask_choice=args.mask_choice,
                           temperature=args.temperature, alpha=args.alpha,
                           seed=args.seed + 1000)

    t_estagio = time.time()
    # No FMA uma epoca tem 800 lotes e leva minutos; reportar so no fim da epoca
    # deixaria longos silencios. Reporta ~10 vezes por epoca -- mas com piso de
    # 10 lotes, senao uma corrida pequena (teste de fumaca) loga a cada lote.
    passo_log = max(10, len(train_dl) // 10)

    for ep in range(ep_inicial, args.msm_epochs + 1):
        t_ep = time.time()
        gerador.manual_seed(args.seed * 1000 + ep)
        model.train()
        tot, n = 0.0, 0
        for i_lote, batch in enumerate(train_dl, 1):
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
            if i_lote % passo_log == 0 and i_lote < len(train_dl):
                log_etapa('MSM %d/%d' % (ep, args.msm_epochs),
                          i_lote, len(train_dl), 'loss=%.4f' % (tot / max(n, 1)))
        train_loss = tot / max(n, 1)

        val_dl = DataLoader(ds['validation'], batch_size=args.batch_size,
                            shuffle=False, collate_fn=val_col(), num_workers=0)
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

        history.append({'epoch': ep, 'msm_train_loss': train_loss,
                        'msm_val_loss': val_loss,
                        'minutos': (time.time() - t_ep) / 60.0})
        # ETA do Estagio 1 pela media das epocas feitas NESTA sessao.
        media_ep = (time.time() - t_estagio) / 60.0 / (ep - ep_inicial + 1)
        log_etapa('ESTAGIO 1', ep, args.msm_epochs,
                  'train=%.4f val=%.4f | resta ~%s'
                  % (train_loss, val_loss, hm(media_ep * (args.msm_epochs - ep))))

        if ep < args.msm_epochs:
            salvar_torch({'model': model.state_dict(), 'opt': opt.state_dict(),
                          'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                          'rng_mascara': train_col.rng.bit_generator.state,
                          'history': history, 'epoca': ep}, ckpt_path)
            motivo = deve_parar()
            if motivo:
                raise PausaSolicitada(motivo)

    salvar_torch(model.ast.state_dict(), enc_path)
    save_json(hist_path, history)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)          # ~1 GB; o encoder final ja esta salvo

    del model, opt
    torch.cuda.empty_cache()
    return enc_path, history


def _msm_sem_checkpoint(strategy, ds, args, tag, ckpt_path):
    """Checkpoint corrompido: descarta e refaz o Estagio 1 do zero."""
    try:
        os.remove(ckpt_path)
    except OSError:
        pass
    torch.cuda.empty_cache()
    return msm_pretrain(strategy, ds, args, tag)


# ---------------------------------------------------------------------------
# Estagio 2: fine-tuning de classificacao em espectrograma LIMPO
# ---------------------------------------------------------------------------

class ProgressoFineTuning(TrainerCallback):
    """Reporta o progresso do Estagio 2 no mesmo formato do Estagio 1.

    O Trainer da HuggingFace tem barra de progresso propria, mas ela depende de
    TTY real: com a saida redirecionada para arquivo de log ela imprime uma
    linha por atualizacao e polui tudo. Este callback reporta uma linha por
    epoca, legivel tanto no terminal quanto no log.
    """

    def __init__(self, total_epocas):
        self.total = total_epocas
        self.inicio = None

    def on_train_begin(self, args, state, control, **kw):
        self.inicio = time.time()
        self.ep0 = int(round(state.epoch or 0))     # > 0 quando retomado

    def on_epoch_end(self, args, state, control, **kw):
        ep = int(round(state.epoch or 0))
        media = (time.time() - self.inicio) / 60.0 / max(ep - self.ep0, 1)
        log_etapa('ESTAGIO 2', ep, self.total,
                  'resta ~%s' % hm(media * (self.total - ep)))


class PausaNoFineTuning(TrainerCallback):
    """Para o Estagio 2 logo DEPOIS de um checkpoint gravado (on_save e chamado
    com o checkpoint ja completo em disco), se pediram pausa."""

    def on_save(self, args, state, control, **kw):
        motivo = deve_parar()
        if motivo and state.epoch is not None and state.epoch < args.num_train_epochs:
            raise PausaSolicitada(motivo)


def permitir_rng_numpy():
    """O Trainer (4.46) grava o estado das RNGs com arrays numpy e o le de
    volta com torch.load, que no PyTorch >= 2.6 usa weights_only=True e recusa
    numpy. Sem isto, TODA retomada no Estagio 2 falha (verificado no ensaio de
    pausa). Libera so os tipos numpy necessarios, nao o pickle inteiro."""
    try:
        nucleo = getattr(np, '_core', None) or np.core
        torch.serialization.add_safe_globals([
            nucleo.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype(np.uint32)), type(np.dtype(np.float64))])
    except Exception as exc:          # torch antigo: nao precisa
        log.debug('add_safe_globals indisponivel: %s', exc)


def remover_rng(ck):
    for nome in os.listdir(ck):
        if nome.startswith('rng_state'):
            os.remove(os.path.join(ck, nome))


def checkpoint_para_retomar(work):
    """Checkpoint do Trainer mais recente que esteja COMPLETO, ou None.

    O Trainer grava o trainer_state.json por ultimo; um checkpoint sem ele foi
    interrompido no meio da gravacao e e descartado."""
    if not os.path.isdir(work):
        return None
    cks = []
    for nome in os.listdir(work):
        if nome.startswith('checkpoint-'):
            d = os.path.join(work, nome)
            if os.path.exists(os.path.join(d, 'trainer_state.json')):
                cks.append((int(nome.split('-')[1]), d))
            else:
                shutil.rmtree(d, ignore_errors=True)
    return max(cks)[1] if cks else None


def finetune(ds, args, tag, dataset, encoder_state=None):
    work = os.path.join(WORK_DIR, tag)
    saida = os.path.join(RUNS_DIR, tag)
    os.makedirs(work, exist_ok=True)
    os.makedirs(saida, exist_ok=True)

    model = fresh_classifier(dataset)
    if encoder_state is not None:
        sd = torch.load(encoder_state, map_location='cpu', weights_only=True)
        missing, unexpected = model.audio_spectrogram_transformer.load_state_dict(
            sd, strict=False)
        log.info('[%s] encoder MSM carregado (missing=%d unexpected=%d)',
                 tag, len(missing), len(unexpected))
        if len(missing) or len(unexpected):
            log.warning('[%s] missing=%s unexpected=%s', tag,
                        list(missing)[:5], list(unexpected)[:5])
    model.to(device_of())

    clean = CleanCollator()
    targs = TrainingArguments(
        output_dir=work,
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
        logging_dir=os.path.join(work, 'runs'),
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to='none',
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        disable_tqdm=True,          # o progresso vem do callback abaixo
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds['train'], eval_dataset=ds['validation'],
        data_collator=clean, compute_metrics=compute_metrics,
        callbacks=[ProgressoFineTuning(args.ft_epochs), PausaNoFineTuning()],
    )
    retomar = checkpoint_para_retomar(work)
    if retomar:
        log.info('[%s] RETOMANDO o Estagio 2 de %s', tag, os.path.basename(retomar))
        permitir_rng_numpy()
    try:
        trainer.train(resume_from_checkpoint=retomar)
    except pickle.UnpicklingError as exc:
        if not retomar:
            raise
        # Ultimo recurso: sem o estado das RNGs a retomada segue valida, so
        # nao reproduz bit a bit a ordem dos lotes das epocas restantes.
        log.warning('[%s] estado das RNGs ilegivel (%s); retomando sem ele.',
                    tag, str(exc).splitlines()[0][:80])
        remover_rng(retomar)
        trainer.train(resume_from_checkpoint=retomar)

    curve = [
        {'epoch': r.get('epoch'), 'eval_loss': r.get('eval_loss'),
         'eval_f1_macro': r.get('eval_f1_macro'),
         'eval_accuracy': r.get('eval_accuracy')}
        for r in trainer.state.log_history if 'eval_loss' in r
    ]

    pred = trainer.predict(ds['test'], metric_key_prefix='test')
    test = pred.metrics

    # Predicoes por exemplo. Sem elas nao ha teste pareado (McNemar) entre
    # estrategias, e os classificadores sao descartados -- esta e a unica chance
    # de guardar essa informacao.
    save_json(os.path.join(saida, 'test_preds.json'), {
        'y_true': [int(x) for x in np.asarray(pred.label_ids).ravel()],
        'y_pred': [int(x) for x in np.asarray(pred.predictions).argmax(-1)],
    })

    res = {
        'test_accuracy': float(test.get('test_accuracy', float('nan'))),
        'test_f1_macro': float(test.get('test_f1_macro', float('nan'))),
        'test_loss': float(test.get('test_loss', float('nan'))),
        'n_test': int(len(ds['test'])),
        'curve': curve,
    }

    del model, trainer
    torch.cuda.empty_cache()
    limpar_checkpoints(work)
    return res


# ---------------------------------------------------------------------------
# Condicoes
# ---------------------------------------------------------------------------

def rodar_e2(strategy, args, ds, tag):
    """MSM (estagio 1) + fine-tuning LIMPO (estagio 2)."""
    log.info('[%s] Estagio 1: pre-treino MSM (%d epocas, mr=%.2f)',
             tag, args.msm_epochs, args.mask_ratio)
    enc, hist = msm_pretrain(strategy, ds, args, tag)
    log.info('[%s] Estagio 1 ok. val_loss inicial=%.4f final=%.4f (queda %.1f%%)',
             tag, hist[0]['msm_val_loss'], hist[-1]['msm_val_loss'],
             100 * (1 - hist[-1]['msm_val_loss'] / max(hist[0]['msm_val_loss'], 1e-12)))

    log.info('[%s] Estagio 2: fine-tuning LIMPO (%d epocas)', tag, args.ft_epochs)
    res = finetune(ds, args, tag, args.dataset, encoder_state=enc)
    res['msm_val_loss_first'] = hist[0]['msm_val_loss']
    res['msm_val_loss_last'] = hist[-1]['msm_val_loss']
    save_json(os.path.join(RUNS_DIR, tag, 'msm_history.json'), hist)

    # O encoder ja foi absorvido pelo classificador; guardar 340 MB por execucao
    # nao serve para nada na analise.
    try:
        os.remove(enc)
        log.info('[%s] encoder MSM removido (economia ~340 MB)', tag)
    except OSError:
        pass
    return res


def rodar_controle(args, ds, tag):
    """Controle: fine-tuning LIMPO sem nenhum MSM.

    Responde 'o MSM ajudou alguma coisa?'. E o piso comum das 4 estrategias --
    sem MSM, a estrategia de mascaramento e irrelevante, entao ha UM controle
    por semente, nao um por estrategia.
    """
    log.info('[%s] controle: fine-tuning LIMPO sem MSM', tag)
    return finetune(ds, args, tag, args.dataset, encoder_state=None)


# ---------------------------------------------------------------------------
# Plano
# ---------------------------------------------------------------------------

def homburg_v1_compativel():
    c = load_json(COMPAT_FILE, {})
    return bool(c.get('homburg_v1_compativel'))


def condicoes_de(dataset, args):
    """[(kind, strategy)] de um bloco de semente.

    Homburg compativel com a v1: so as estrategias novas (controle e as 4
    originais ja existem, rodadas com o mesmo conjunto de teste). Caso
    contrario, e no FMA: o bloco completo, controle + todas as estrategias.
    """
    estrategias = list(args.strategies)
    if (dataset == 'homburg' and not args.homburg_completo
            and not args.fumaca and homburg_v1_compativel()):
        return [('e2', s) for s in estrategias if s in ESTRATEGIAS_NOVAS]
    return [('ctl', 'noadapt')] + [('e2', s) for s in estrategias]


def montar_plano(args):
    """Lista de (dataset, seed, kind, strategy), em ordem de bloco de semente."""
    jobs = []
    for dataset in args.datasets:
        conds = condicoes_de(dataset, args)
        for seed in args.seeds:
            for kind, s in conds:
                jobs.append((dataset, seed, kind, s))
    return jobs


def chave(dataset, seed, kind, strategy):
    return '%s|%s|%s|s%d' % (dataset, kind, strategy, seed)


def tag_de(dataset, seed, kind, strategy):
    return '%s-%s-%s-s%d' % (dataset, kind, strategy, seed)


def imprimir_plano(args, state):
    jobs = montar_plano(args)
    feitos = sum(1 for j in jobs if state.get(chave(*j), {}).get('done'))
    print()
    print('PLANO -- %d execucoes (%d ja concluidas, %d restantes)'
          % (len(jobs), feitos, len(jobs) - feitos))
    print()
    print('  Ordem: bloco completo de semente (controle + %d estrategias).'
          % len(args.strategies))
    if 'homburg' in args.datasets and homburg_v1_compativel() and not args.homburg_completo:
        print('  Homburg: so as estrategias novas -- controle e as 4 originais')
        print('  sao reaproveitados da corrida anterior (mesmo conjunto de teste).')
    print('  Cada bloco concluido ja e utilizavel na analise estatistica.')
    print()
    for dataset in args.datasets:
        n_train = {'homburg': 1319, 'fma': 6400}.get(dataset, '?')
        n_test = {'homburg': 285, 'fma': 800}.get(dataset, '?')
        do_ds = [j for j in jobs if j[0] == dataset]
        pend = [j for j in do_ds if not state.get(chave(*j), {}).get('done')]
        print('  %-8s treino~%s  teste~%s  %d execucoes (%d pendentes)'
              % (dataset, n_train, n_test, len(do_ds), len(pend)))
    print()
    # Estimativa por dataset e por tipo (controle e bem mais barato), pela
    # mediana medida nesta maquina. Sem medicao do FMA, escala a do Homburg
    # pelo tamanho do treino (6400 / 1319).
    medidos = [r for r in load_json(RESULTS_FILE, {}).values() if r.get('minutos')]

    def mediana(ds_, kind):
        v = [r['minutos'] for r in medidos
             if r['dataset'] == ds_ and r['condicao'] == kind]
        return float(np.median(v)) if v else None

    total_min, ok_est = 0.0, True
    for j in jobs:
        if state.get(chave(*j), {}).get('done'):
            continue
        m = mediana(j[0], j[2])
        if m is None and j[0] == 'fma':
            mh = mediana('homburg', j[2])
            m = mh * 6400 / 1319.0 if mh else None
        if m is None:
            ok_est = False
            break
        total_min += m
    if ok_est and medidos:
        print('  Estimativa para o que falta, pelos tempos medidos aqui: %.0f h (~%.1f dias)'
              % (total_min / 60.0, total_min / 60.0 / 24))
    else:
        print('  Sem medicao suficiente ainda -- a estimativa aparece depois das')
        print('  primeiras execucoes (e o verificar_ambiente.py da uma ordem de grandeza).')
    print()


def arguments():
    p = argparse.ArgumentParser(
        description='Orquestrador da execucao remota (FMA-small + Homburg).')
    p.add_argument('--datasets', nargs='+', default=PLANO_DATASETS,
                   choices=sorted(DATASETS))
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS_PADRAO)
    p.add_argument('--strategies', nargs='+', default=STRATEGIES)
    p.add_argument('--plano', action='store_true',
                   help='Imprime o plano e a estimativa de tempo, sem rodar nada.')
    p.add_argument('--fumaca', action='store_true',
                   help='Teste de fumaca: escala minima, prova o pipeline inteiro.')
    p.add_argument('--parar-as', default=None, metavar='HH:MM',
                   help='Pausa sozinho (salvando) ao passar deste horario.')
    p.add_argument('--homburg-completo', action='store_true',
                   help='Roda o Homburg inteiro (controle + 6 estrategias) mesmo '
                        'que a corrida anterior seja reaproveitavel.')
    p.add_argument('--max-examples', type=int, default=None,
                   help='Limita exemplos por split (implicado por --fumaca).')
    # Hiperparametros -- travados nos do GTZAN. Nao alterar sem registrar o motivo.
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
    return p.parse_args()


def pausar(motivo, args):
    log.info('=' * 72)
    log.info('PAUSADO: %s.', motivo)
    log.info('O progresso foi salvo ao fim da ultima epoca.')
    if os.path.exists(ARQUIVO_PAUSA):
        try:
            os.remove(ARQUIVO_PAUSA)     # senao a proxima sessao pararia na hora
        except OSError:
            pass
    if not args.fumaca:
        try:
            import coletar_resultados
            coletar_resultados.coletar(silencioso=True)
            log.info('Resultados ate aqui empacotados em %s.', coletar_resultados.SAIDA)
        except Exception as exc:
            log.warning('Falha ao empacotar: %s', exc)
    log.info('Para continuar, rode o MESMO comando de novo.')
    log.info('=' * 72)


def main():
    args = arguments()

    if args.fumaca:
        args.seeds = [42]
        args.max_examples = args.max_examples or 40
        args.msm_epochs = 2
        args.ft_epochs = 2
        setup_logging('fumaca.log')
    else:
        setup_logging()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    state_file = STATE_FILE if not args.fumaca else STATE_FILE.replace('.json', '_fumaca.json')
    results_file = RESULTS_FILE if not args.fumaca else RESULTS_FILE.replace('.json', '_fumaca.json')
    state = load_json(state_file, {})
    results = load_json(results_file, {})

    if args.plano:
        imprimir_plano(args, state)
        return

    env = ambiente()
    save_json(os.path.join(RESULTS_DIR, 'ambiente.json'), env)

    log.info('=' * 72)
    log.info('run_remoto %s', 'TESTE DE FUMACA' if args.fumaca else '(corrida real)')
    log.info('GPU: %s (%s GB) | torch %s | cuda %s',
             env.get('gpu', 'NENHUMA'), env.get('vram_gb', '-'),
             env['torch'], env.get('cuda', '-'))
    log.info('datasets=%s sementes=%s estrategias=%s',
             args.datasets, args.seeds, args.strategies)
    log.info('mask_ratio=%.2f temperature=%.3f alpha=%.2f batch=%d '
             'msm_epochs=%d ft_epochs=%d',
             args.mask_ratio, args.temperature, args.alpha, args.batch_size,
             args.msm_epochs, args.ft_epochs)
    log.info('disco livre: %.1f GB', free_gb())
    log.info('=' * 72)

    configurar_parada(args.parar_as)
    if os.path.exists(ARQUIVO_PAUSA):
        os.remove(ARQUIVO_PAUSA)         # pedido de pausa de uma sessao antiga

    if not torch.cuda.is_available():
        log.warning('CUDA NAO disponivel -- vai rodar na CPU, o que e '
                    'inviavelmente lento. Verifique a instalacao do torch.')

    jobs = montar_plano(args)
    pendentes = [j for j in jobs if not state.get(chave(*j), {}).get('done')]
    log.info('%d execucoes no plano, %d pendentes.', len(jobs), len(pendentes))

    cache_carregado = {}
    t_inicio = time.time()
    n_feitos = 0
    PROG.update({'total': len(pendentes), 'feitos': 0, 'inicio': t_inicio})

    for dataset, seed, kind, strategy in jobs:
        k = chave(dataset, seed, kind, strategy)
        if state.get(k, {}).get('done'):
            continue
        if free_gb() < MIN_FREE_GB:
            log.error('Disco abaixo de %.1f GB -- abortando para nao corromper '
                      'nada. Libere espaco e reinvoque o script.', MIN_FREE_GB)
            break

        if dataset not in cache_carregado:
            log.info('Carregando cache de %s ...', dataset)
            cache_carregado[dataset] = carregar_dataset(dataset, args.max_examples)
            n = {s: len(d) for s, d in cache_carregado[dataset].items()}
            log.info('  %s: %s', dataset, n)
        ds = cache_carregado[dataset]

        args.dataset = dataset
        args.seed = seed
        tag = tag_de(dataset, seed, kind, strategy)

        log.info('-' * 72)
        log.info('INICIANDO %s   (execucao %d de %d)', k, n_feitos + 1, len(pendentes))
        if n_feitos == 0:
            log_geral()          # depois disso, a linha GERAL sai a cada conclusao
        t0 = time.time()
        try:
            if kind == 'ctl':
                res = rodar_controle(args, ds, tag)
            else:
                res = rodar_e2(strategy, args, ds, tag)
        except PausaSolicitada as motivo:
            pausar(motivo, args)
            sys.exit(3)
        except KeyboardInterrupt:
            log.info('')
            log.info('INTERROMPIDO (Ctrl+C). Nada se perde alem da epoca em andamento.')
            log.info('Para continuar, rode o MESMO comando de novo.')
            if not args.fumaca:
                empacotar_sem_falhar()
            sys.exit(130)
        except Exception as exc:
            log.exception('[FALHA] %s: %s', k, exc)
            state[k] = {'done': False, 'error': str(exc)}
            save_json(state_file, state)
            torch.cuda.empty_cache()
            continue

        res.update({
            'dataset': dataset, 'condicao': kind, 'estrategia': strategy,
            'semente': seed,
            'minutos': (time.time() - t0) / 60.0,
            'quando': datetime.datetime.now().isoformat(timespec='seconds'),
        })
        results[k] = res
        state[k] = {'done': True}
        save_json(results_file, results)
        save_json(state_file, state)
        save_json(os.path.join(RUNS_DIR, tag, 'result.json'), res)
        # Rascunho desta execucao (checkpoints, encoder) ja nao serve para nada.
        shutil.rmtree(os.path.join(WORK_DIR, tag), ignore_errors=True)

        n_feitos += 1
        PROG['feitos'] = n_feitos
        log.info('CONCLUIDO %s em %s | acc=%.4f f1=%.4f',
                 k, hm(res['minutos']), res['test_accuracy'], res['test_f1_macro'])
        log_geral()
        log.info('         disco livre %.1f GB', free_gb())

        # Fim de um bloco de semente: reempacota os resultados. Assim existe
        # SEMPRE um zip atualizado para enviar, mesmo que a maquina caia ou a
        # corrida tenha de ser interrompida antes do fim.
        bloco = [j for j in jobs if j[0] == dataset and j[1] == seed]
        if not args.fumaca and all(state.get(chave(*j), {}).get('done') for j in bloco):
            try:
                import coletar_resultados
                coletar_resultados.coletar(silencioso=True)
                log.info('Bloco %s semente %d completo -> %s atualizado.',
                         dataset, seed, coletar_resultados.SAIDA)
            except Exception as exc:          # empacotar nunca derruba a corrida
                log.warning('Falha ao empacotar (a corrida segue): %s', exc)

    # O rascunho de cada execucao concluida ja foi apagado; o que sobrar em
    # trabalho_tmp/ e de execucoes por retomar (ou que falharam) e fica.
    if os.path.isdir(WORK_DIR) and not os.listdir(WORK_DIR):
        shutil.rmtree(WORK_DIR, ignore_errors=True)

    falhas = [chave(*j) for j in jobs if not state.get(chave(*j), {}).get('done')]
    log.info('=' * 72)
    log.info('Fim. %d execucoes novas. Resultados em %s', n_feitos, results_file)
    if falhas and not args.fumaca:
        log.error('%d execucao(oes) NAO concluida(s): %s', len(falhas), falhas[:6])
        log.error('Rode de novo para tentar outra vez. Se repetir, envie o zip '
                  '(contem os logs com o erro).')
        try:
            import coletar_resultados
            coletar_resultados.coletar(silencioso=True)
        except Exception:
            pass
        sys.exit(4)
    if args.fumaca:
        falhas = [k for k, v in state.items() if not v.get('done')]
        if falhas:
            log.error('Teste de fumaca com FALHAS: %s', falhas)
            sys.exit(1)
        log.info('Teste de fumaca OK.')
    else:
        log.info('Gere a tabela com: python analisar_remoto.py')
        log.info('Empacote para enviar com: python coletar_resultados.py')


def empacotar_sem_falhar():
    """Regrava RESULTADOS_PARA_ENVIAR.zip; nunca levanta excecao."""
    try:
        import coletar_resultados
        coletar_resultados.coletar(silencioso=True)
    except BaseException:            # inclui o sys.exit do coletar sem resultados
        pass


if __name__ == '__main__':
    # Qualquer saida que nao seja a normal (erro inesperado, Ctrl+C fora de
    # uma execucao) ainda deixa o zip de resultados/logs atualizado.
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        if '--fumaca' not in sys.argv and '--plano' not in sys.argv:
            empacotar_sem_falhar()
        sys.exit(130)
    except BaseException as exc:
        log.exception('ERRO INESPERADO: %s', exc)
        if '--fumaca' not in sys.argv and '--plano' not in sys.argv:
            empacotar_sem_falhar()
        sys.exit(5)
