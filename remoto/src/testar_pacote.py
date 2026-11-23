"""
testar_pacote.py
================
Validacao do pacote SEM GPU, SEM download e SEM dataset. Roda em segundos.

    python testar_pacote.py

Se algum teste falhar, NAO comece a corrida -- avise quem enviou o pacote.

Estes testes existem porque o projeto perdeu os resultados de dois datasets por
um bug de remontagem de patches que passou meses sem ser detectado: o pipeline
entregava ao modelo os valores certos em posicoes erradas (0% de acerto
posicional). O teste 2 e a rede de seguranca contra exatamente isso.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

falhas = []


def checar(nome, condicao, detalhe=''):
    if condicao:
        print('  [ok]    %s' % nome)
    else:
        print('  [FALHA] %s %s' % (nome, detalhe))
        falhas.append(nome)


def patches_realistas(rng, n=496):
    """Patches com espalhamento de energia parecido com o de um espectrograma
    real: bandas graves fortes, agudos fracos, alguns transientes.

    Ruido gaussiano puro NAO serve para testar seletividade -- a energia RMS de
    256 amostras N(0,1) e ~1.00 +- 0.04, entao todos os patches ficam
    praticamente empatados e qualquer estrategia parece quase aleatoria.
    """
    escala = np.exp(rng.normal(0.0, 1.2, size=n))          # energia log-normal
    p = rng.normal(size=(n, 16, 16)) * escala[:, None, None]
    transientes = rng.random(n) < 0.15
    p[transientes] += np.linspace(0, 4, 16)[None, None, :]  # rampa temporal
    return p.astype(np.float32)


# ---------------------------------------------------------------------------
print()
print('1. Scores das 4 estrategias')

from masking.patch_scoring import SpectralEnergyScoring, OnsetScoring, CombinedScoring
from remoto_common import combinar_scores, scores_for, STORED_ALPHA

rng = np.random.default_rng(0)

# 1a) A recombinacao e matematicamente identica a classe original, em precisao
#     plena. E o que autoriza usar alphas fora do valor canonico.
div = 0
for trial in range(200):
    patches = patches_realistas(rng)
    if trial % 7 == 0:
        patches[:] = 0.0                    # silencio total
    if trial % 11 == 0:
        patches[:] = 3.5                    # constante (minmax degenerado)
    if trial % 13 == 0:
        patches *= 1e-6                     # quase-silencio
    e = SpectralEnergyScoring().score_patches(patches, normalize=False)
    o = OnsetScoring().score_patches(patches, normalize=False)
    for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
        ref = CombinedScoring(alpha=alpha).score_patches(patches, normalize=False)
        if not np.array_equal(ref, combinar_scores(e, o, alpha)):
            div += 1
checar('combinar_scores == CombinedScoring, bit-a-bit (5 alphas x 200 casos)',
       div == 0, '(%d divergencias de 1000)' % div)
checar('dtypes dos scorers originais sao os esperados (float32 / float64)',
       e.dtype == np.float32 and o.dtype == np.float64,
       '(energy=%s onset=%s)' % (e.dtype, o.dtype))


# ---------------------------------------------------------------------------
print()
print('2. REGRESSAO DO BUG DE REMONTAGEM (o que invalidou FMA e Homburg antes)')

from masking.spectrogram_utils import (
    patches_to_spectrogram, patches_to_ast_input, ast_input_to_patches,
)

grid = (8, 62)

# Espectrograma com um valor UNICO por posicao: se a remontagem embaralhar
# qualquer coisa, a comparacao acusa.
espec = np.arange(128 * 992, dtype=np.float32).reshape(128, 992)
p = espec.reshape(8, 16, 62, 16).transpose(0, 2, 1, 3).reshape(-1, 16, 16)
pt = torch.from_numpy(p).unsqueeze(0)

volta = patches_to_spectrogram(pt, grid)[0].numpy()
acerto = float((volta == espec).mean())
checar('patches -> espectrograma recupera 100% das posicoes', acerto == 1.0,
       '(acerto posicional = %.1f%%)' % (100 * acerto))

ast_in = patches_to_ast_input(pt, grid, target_frames=1024)
checar('layout do AST e (B, 1024, 128) = (tempo, frequencia)',
       tuple(ast_in.shape) == (1, 1024, 128), '(obtido %s)' % (tuple(ast_in.shape),))
checar('padding temporal preenche com zero apos o fim do sinal',
       bool((ast_in[0, 992:, :] == 0).all()))
checar('ida e volta patches -> AST -> patches e exata',
       bool(torch.equal(ast_input_to_patches(ast_in, grid, patch_size=16), pt)))

# A forma bugada antiga, registrada para deixar claro que ela errava tudo.
bugado = pt.reshape(1, -1, 128)[:, :992, :].transpose(1, 2)[0].numpy()
acerto_bug = float((bugado == espec).mean())
checar('a remontagem ANTIGA de fato errava (esperado ~0%)', acerto_bug < 0.01,
       '(acerto do metodo antigo = %.2f%%)' % (100 * acerto_bug))


# ---------------------------------------------------------------------------
print()
print('3. Cache: schema, custo em disco e fidelidade ao que o GTZAN consumia')

from datasets import Dataset
import preprocess_remoto as prep

checar('grade calculada dos parametros e (8, 62) = 496 patches',
       prep.GRID == (8, 62) and prep.N_PATCHES == 496,
       '(obtido %s / %d)' % (prep.GRID, prep.N_PATCHES))

wav = rng.normal(size=int(prep.SEGMENT_SECS * prep.SAMPLE_RATE)).astype(np.float32)
rec = prep.make_record(wav, 3, random_crop=False, rng=np.random.default_rng(1))
checar('make_record devolve patches (496, 16, 16) float32',
       rec['input_values'].shape == (496, 16, 16)
       and rec['input_values'].dtype == np.float32,
       '(%s %s)' % (rec['input_values'].shape, rec['input_values'].dtype))
checar('make_record grava as tres colunas de score',
       {'score_energy', 'score_onset', 'score_combined'} <= set(rec))

tmp = Path(tempfile.mkdtemp(prefix='teste_cache_'))
try:
    ds = Dataset.from_generator(
        lambda: (rec for _ in range(4)), features=prep.features_schema())
    ds.save_to_disk(str(tmp / 'split'))
    lido = Dataset.load_from_disk(str(tmp / 'split')).with_format('numpy')
    e0 = lido[0]

    checar('patches sobrevivem ao disco sem perda',
           np.array_equal(e0['input_values'], rec['input_values']))
    checar('dtype dos patches no disco e float32 (metade do espaco)',
           e0['input_values'].dtype == np.float32,
           '(obtido %s)' % e0['input_values'].dtype)

    # O ponto central: cada estrategia recebe no Estagio 1 exatamente
    # float64(float32(saida_do_scorer)) -- que e o que o collator do GTZAN
    # recebia, porque `datasets` rebaixa para float32 na leitura e o collator
    # antigo fazia o upcast de volta.
    esperado = {
        'spectral_energy': rec['score_energy'],
        'onset':           rec['score_onset'],
        'combined':        rec['score_combined'],
    }
    for estrategia, bruto in esperado.items():
        alvo = np.asarray(np.asarray(bruto, dtype=np.float32), dtype=np.float64)
        obtido = scores_for(estrategia, e0)
        checar('%-16s recebe float64(float32(scorer)), como no GTZAN' % estrategia,
               np.array_equal(obtido, alvo),
               '(maxdiff %.3e)' % np.abs(obtido - alvo).max())
    checar('random nao usa score nenhum', scores_for('random', e0) is None)

    # Alpha nao-canonico: derivado das componentes ja arredondadas. Registra o
    # tamanho do desvio para que ninguem se surpreenda depois.
    derivado = scores_for('combined', {k: v for k, v in e0.items()
                                       if k != 'score_combined'}, alpha=STORED_ALPHA)
    desvio = float(np.abs(derivado - scores_for('combined', e0)).max())
    checar('derivar combined em vez de ler a coluna desvia < 1e-6',
           desvio < 1e-6, '(desvio %.3e)' % desvio)
    print('           -> desvio da derivacao: %.2e (epsilon do float32 = %.2e)'
          % (desvio, np.finfo(np.float32).eps))

    kb = sum(f.stat().st_size for f in (tmp / 'split').rglob('*')
             if f.is_file()) / len(lido) / 1024
    checar('custo por exemplo abaixo de 700 KB', kb < 700, '(medido %.0f KB)' % kb)
    print('           -> %.0f KB/exemplo; FMA 8000 = %.1f GB, Homburg 1886 = %.1f GB, total %.1f GB'
          % (kb, kb * 8000 / 1024 ** 2, kb * 1886 / 1024 ** 2,
             kb * 9886 / 1024 ** 2))
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
print()
print('4. Collators: proporcao mascarada, seletividade e formato')

from remoto_common import CleanCollator, MSMCollator

exemplos = []
for i in range(8):
    patches = patches_realistas(rng)
    exemplos.append({
        'input_values': patches,
        'score_energy': SpectralEnergyScoring().score_patches(patches, normalize=False),
        'score_onset': OnsetScoring().score_patches(patches, normalize=False),
        'score_combined': CombinedScoring(alpha=STORED_ALPHA).score_patches(
            patches, normalize=False),
        'label': i % 8,
        'grid_shape': np.array([8, 62]),
    })

b = CleanCollator()(exemplos)
checar('CleanCollator devolve input_values (8, 1024, 128)',
       tuple(b['input_values'].shape) == (8, 1024, 128),
       '(obtido %s)' % (tuple(b['input_values'].shape),))
checar('CleanCollator devolve rotulos long', b['labels'].dtype == torch.long)
checar('CleanCollator NAO zera nada (fine-tuning e em espectrograma limpo)',
       bool((b['input_values'][0, :992, :] != 0).any()))

from remoto_common import STRATEGIES
for estrategia in STRATEGIES:
    out = MSMCollator(estrategia, mask_ratio=0.75, seed=42)(exemplos)
    frac = out['mask'].float().mean().item()
    checar('%-16s mascara 75%% dos patches' % estrategia,
           abs(frac - 0.75) < 0.01, '(medido %.3f)' % frac)
    checar('%-16s formato do alvo de reconstrucao e (8, 496, 16, 16)' % estrategia,
           tuple(out['clean_patches'].shape) == (8, 496, 16, 16),
           '(obtido %s)' % (tuple(out['clean_patches'].shape),))
    zerado = (out['input_values'][0, :992, :] == 0).float().mean().item()
    checar('%-16s patches mascarados aparecem zerados na entrada' % estrategia,
           zerado > 0.5, '(fracao zerada %.2f)' % zerado)
    checar('%-16s alvo preserva o conteudo ORIGINAL (nao o mascarado)' % estrategia,
           np.allclose(out['clean_patches'][0].numpy(), exemplos[0]['input_values']))


def correlacao(estrategia):
    """Correlacao entre o score da estrategia e a mascara que ela sorteou."""
    out = MSMCollator(estrategia, mask_ratio=0.5, seed=7)(exemplos)
    ref = 'spectral_energy' if estrategia == 'random' else estrategia
    cors = [np.corrcoef(scores_for(ref, ex), out['mask'][i].numpy().astype(float))[0, 1]
            for i, ex in enumerate(exemplos)]
    return float(np.mean(cors))


c = {e: correlacao(e) for e in ('random', 'spectral_energy', 'onset', 'combined')}
checar('random NAO correlaciona mascara com score (baseline uniforme)',
       abs(c['random']) < 0.15, '(correlacao %+.3f)' % c['random'])
for e in ('spectral_energy', 'onset', 'combined'):
    # Limiar em 0.40: as tres medem ~0.50 nesta fixture e o baseline uniforme
    # mede ~0.00. A correlacao e uma estatistica estocastica (depende da semente
    # do sorteio), entao o limiar precisa de folga em vez de encostar no valor
    # observado -- o que o teste tem de detectar e uma estrategia que deixou de
    # ser seletiva, nao uma flutuacao de terceira casa.
    checar('%-16s concentra a mascara nos patches de score alto' % e,
           c[e] > 0.40, '(correlacao %+.3f)' % c[e])
print('           -> correlacao score/mascara: ' +
      ' | '.join('%s %+.2f' % (k, v) for k, v in c.items()))


# ---------------------------------------------------------------------------
print()
print('4b. Estrategias novas (inverse_energy, specaugment) e regressao das antigas')

from masking.patch_mask_collator import (
    select_masked, weighted_random_order, specaugment_indices)

# As 4 originais precisam consumir a RNG EXATAMENTE como antes da refatoracao,
# senao as mascaras deixam de ser as das corridas anteriores.
iguais = True
for estrategia, chave_s in (('random', None), ('spectral_energy', 'score_energy'),
                            ('onset', 'score_onset'), ('combined', 'score_combined')):
    r_novo, r_velho = np.random.default_rng(5), np.random.default_rng(5)
    for ex in exemplos:
        sc = None if chave_s is None else np.asarray(ex[chave_s], dtype=np.float64)
        novo = select_masked(estrategia, sc, 496, 0.75, r_novo, (8, 62))
        if chave_s is None:
            velho = r_velho.permutation(496)[:372]
        else:
            velho = weighted_random_order(sc, r_velho, 0.1)[:372]
        iguais &= bool(np.array_equal(novo, velho))
checar('as 4 estrategias originais sorteiam as MESMAS mascaras de antes', iguais)

out = MSMCollator('inverse_energy', mask_ratio=0.5, seed=7)(exemplos)
c_inv = float(np.mean([np.corrcoef(scores_for('spectral_energy', ex),
                                   out['mask'][i].numpy().astype(float))[0, 1]
                       for i, ex in enumerate(exemplos)]))
checar('inverse_energy mascara os patches de MENOR energia (correlacao negativa)',
       c_inv < -0.40, '(correlacao %+.3f)' % c_inv)

def frac_em_faixas(idx):
    """Fracao dos patches mascarados que estao numa coluna (tempo) ou linha
    (frequencia) INTEIRAMENTE mascarada."""
    m = np.zeros((8, 62), dtype=bool)
    m.flat[idx] = True
    cheias = m.all(axis=0)[None, :] | m.all(axis=1)[:, None]
    return float((m & cheias).sum() / m.sum())


rng_sa = np.random.default_rng(3)
amostras_sa = [specaugment_indices((8, 62), 372, rng_sa) for _ in range(20)]
checar('specaugment mascara exatamente k patches distintos',
       all(len(i) == 372 and len(set(i.tolist())) == 372 for i in amostras_sa))
# Estrutura: a maior parte dos patches mascarados cai em faixas inteiras (a
# excecao e a ultima faixa, truncada). No sorteio uniforme isso e ~0,1.
f_sa = float(np.mean([frac_em_faixas(i) for i in amostras_sa]))
f_rd = float(np.mean([frac_em_faixas(rng_sa.permutation(496)[:372]) for _ in range(20)]))
checar('specaugment forma faixas inteiras de tempo/frequencia (random nao)',
       f_sa >= 0.70 and f_rd <= 0.30, '(specaugment %.2f, random %.2f)' % (f_sa, f_rd))
print('           -> fracao em faixas inteiras: specaugment %.2f | random %.2f' % (f_sa, f_rd))
out = MSMCollator('specaugment', mask_ratio=0.75, seed=7)(exemplos)
checar('specaugment nao depende de score (mascaras iguais com scores trocados)',
       bool(torch.equal(out['mask'],
                        MSMCollator('specaugment', mask_ratio=0.75, seed=7)(
                            [dict(e, score_energy=e['score_onset']) for e in exemplos])['mask'])))


# ---------------------------------------------------------------------------
print()
print('5. Reprodutibilidade do crop aleatorio (retomada precisa ser exata)')

r = lambda s, sp, i: prep.example_rng(s, sp, i).integers(0, 10 ** 9)
checar('mesma (semente, split, indice) -> mesmo crop', r(42, 'train', 100) == r(42, 'train', 100))
checar('indices diferentes -> crops diferentes', r(42, 'train', 100) != r(42, 'train', 101))
checar('sementes diferentes -> crops diferentes', r(42, 'train', 100) != r(43, 'train', 100))
checar('splits diferentes -> crops diferentes', r(42, 'train', 100) != r(42, 'test', 100))


# ---------------------------------------------------------------------------
print()
print('6. Registro dos datasets e do plano')

from remoto_common import DATASETS, genres_of
import run_remoto

checar('FMA-small tem 8 generos', len(genres_of('fma')) == 8)
checar('Homburg tem 9 generos', len(genres_of('homburg')) == 9)
checar('Homburg divide 70/15/15 (mesma proporcao do GTZAN)',
       DATASETS['homburg']['split'] == (0.70, 0.15))
checar('FMA usa os splits nativos do repositorio',
       DATASETS['fma']['split'] == 'nativo')

class _A:
    datasets = ['homburg', 'fma']
    seeds = [42, 43, 44, 45, 46, 47]
    strategies = STRATEGIES
    homburg_completo = False
    fumaca = False

import json as _json
_compat_existia = os.path.exists(run_remoto.COMPAT_FILE)
_compat_backup = open(run_remoto.COMPAT_FILE, encoding='utf-8').read() if _compat_existia else None
try:
    os.makedirs(os.path.dirname(run_remoto.COMPAT_FILE), exist_ok=True)
    with open(run_remoto.COMPAT_FILE, 'w', encoding='utf-8') as f:
        _json.dump({'homburg_v1_compativel': False}, f)
    plano = run_remoto.montar_plano(_A())
    checar('plano completo tem 84 execucoes (2 datasets x 6 sementes x 7 condicoes)',
           len(plano) == 84, '(obtido %d)' % len(plano))
    checar('Homburg vem antes do FMA (o barato completa primeiro)',
           plano[0][0] == 'homburg' and plano[-1][0] == 'fma')
    checar('cada bloco e uma semente COMPLETA (controle + 6 estrategias)',
           len({j[1] for j in plano[:7]}) == 1
           and [j[2] for j in plano[:7]] == ['ctl'] + ['e2'] * 6,
           '(obtido %s)' % ([(j[2], j[3]) for j in plano[:7]],))
    checar('chaves do estado sao unicas (nada se sobrescreve)',
           len({run_remoto.chave(*j) for j in plano}) == len(plano))

    with open(run_remoto.COMPAT_FILE, 'w', encoding='utf-8') as f:
        _json.dump({'homburg_v1_compativel': True}, f)
    plano = run_remoto.montar_plano(_A())
    hb = [j for j in plano if j[0] == 'homburg']
    checar('Homburg reaproveitado: so as 2 novas x 6 sementes = 12 execucoes',
           len(hb) == 12 and {j[3] for j in hb} == {'inverse_energy', 'specaugment'},
           '(obtido %d)' % len(hb))
    checar('FMA continua completo: 6 sementes x 7 condicoes = 42',
           len(plano) - len(hb) == 42, '(obtido %d)' % (len(plano) - len(hb)))
finally:
    if _compat_existia:
        with open(run_remoto.COMPAT_FILE, 'w', encoding='utf-8') as f:
            f.write(_compat_backup)
    else:
        os.remove(run_remoto.COMPAT_FILE)


# ---------------------------------------------------------------------------
print()
print('=' * 70)
if falhas:
    print('%d TESTE(S) FALHARAM:' % len(falhas))
    for f in falhas:
        print('  - %s' % f)
    print()
    print('NAO inicie a corrida. Avise quem enviou o pacote.')
    sys.exit(1)
print('TODOS OS TESTES PASSARAM. O pacote esta consistente.')
print()
print('Proximo passo: siga o LEIA-ME.md')
print('=' * 70)
