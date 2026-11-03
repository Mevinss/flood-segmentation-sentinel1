"""Build self-contained notebooks from the reviewed package; no external code host."""
import json
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).parent
sources={str(p.relative_to(ROOT)).replace('\\','/'):p.read_text(encoding='utf-8') for p in (ROOT/'flood').glob('*.py')}
sources['requirements-colab.txt']=(ROOT/'requirements-colab.txt').read_text(encoding='utf-8')
for p in (ROOT/'tests').glob('test_*.py'): sources[str(p.relative_to(ROOT)).replace('\\','/')]=p.read_text(encoding='utf-8')
bootstrap='''from pathlib import Path
import os, sys, json, subprocess
WORK = Path('/content/flood_project')
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
if str(WORK) not in sys.path: sys.path.insert(0, str(WORK))
SOURCES = '''+repr(sources)+'''
for name, source in SOURCES.items():
    dest = WORK / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source, encoding='utf-8')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'])
import torch
from flood.train import environment
print(environment(WORK / 'environment'))
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU — audit only; select GPU before training')
'''
storage='''from google.colab import drive
drive.mount('/content/drive')
SAVE = Path('/content/drive/MyDrive/FloodSegmentation')
SAVE.mkdir(parents=True, exist_ok=True)
(SAVE / 'write_check.txt').write_text('Storage available', encoding='utf-8')
assert (SAVE / 'write_check.txt').read_text(encoding='utf-8') == 'Storage available'
environment(SAVE / 'environment')
print('Persistent results:', SAVE)
'''
data='''from flood.data import prepare, audit, eda
DATA = WORK / 'data'
df = prepare(DATA, include_jrc=True)
assert len(df) == 446, f'Unexpected manifest size: {len(df)} — inspect before training'
norm, stats = audit(df, SAVE / 'data_audit')
df.to_csv(SAVE / 'data_audit' / 'manifest.csv', index=False)
eda(df, SAVE / 'data_audit')
display(df.groupby('split').size().rename('tiles').to_frame())
display(stats.groupby(['split', 'event'])[['valid_pixels','water_pixels']].sum())
print('Train-only normalization:', norm)
'''
tests='''subprocess.check_call([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
'''
preview='''from IPython.display import display, Image
for p in sorted((SAVE / 'data_audit').glob('*_data.png'))[:4]:
    display(Image(filename=str(p)))
'''
smoke='''assert torch.cuda.is_available(), 'Runtime → Change runtime type → GPU'
from flood.train import overfit_check
check_path = SAVE / 'checks'
check_path.mkdir(exist_ok=True)
print(overfit_check(df, norm, check_path, steps=80))
'''
training='''from flood.train import train, DEFAULT
CONFIG = {**DEFAULT, 'batch_size': 8, 'workers': 2, 'epochs': 100}
# Reduce batch_size to 4 or 2 if the first run runs out of GPU memory.
# Keep a separate run name when changing the experiment configuration.
RUN = SAVE / 'runs' / 'resnet34_seed42'
checkpoint = train(df, norm, RUN, CONFIG, resume=True)
print('Best checkpoint:', checkpoint)
'''
evaluation='''from flood.evaluate import evaluate, learning_curves
RUN = SAVE / 'runs' / 'resnet34_seed42'
checkpoint = RUN / 'best.pt'
assert checkpoint.exists(), 'Train the main model first'
main_metrics = evaluate(df, SAVE / 'evaluation' / 'resnet34', checkpoint)
learning_curves(RUN)
print(json.dumps(main_metrics, indent=2))
display(Image(filename=str(RUN / 'learning_curves.png')))
'''
baseline='''from flood.evaluate import evaluate
otsu_metrics = evaluate(df, SAVE / 'evaluation' / 'otsu')
print(json.dumps(otsu_metrics, indent=2))
'''
plain='''from flood.train import train, DEFAULT
from flood.evaluate import evaluate, learning_curves
plain_run = SAVE / 'runs' / 'plain_seed42'
plain_checkpoint = train(df, norm, plain_run, {**DEFAULT, 'model':'plain', 'pretrained':False}, resume=True)
plain_metrics = evaluate(df, SAVE / 'evaluation' / 'plain', plain_checkpoint)
learning_curves(plain_run)
print(json.dumps(plain_metrics, indent=2))
'''
maps='''from flood.evaluate import flood_maps
eval_dir = SAVE / 'evaluation' / 'resnet34'
threshold = json.loads((eval_dir / 'threshold.json').read_text())['threshold']
areas = flood_maps(df, eval_dir, SAVE / 'maps', threshold)
display(areas.head())
print('These are estimates of water outside JRC permanent water, not independently verified flood labels.')
'''
summary='''import pandas as pd
rows = []
for name in ['otsu','plain','resnet34']:
    p = SAVE / 'evaluation' / name / 'metrics.json'
    if not p.exists():
        print('Not completed:', name)
        continue
    for split, values in json.loads(p.read_text()).items():
        rows.append({'model':name, 'split':split, **values})
comparison = pd.DataFrame(rows)
comparison.to_csv(SAVE / 'comparison.csv', index=False)
display(comparison)
for p in sorted((SAVE / 'evaluation' / 'resnet34').glob('*_comparison.png'))[:5]:
    display(Image(filename=str(p)))
print('Results are saved in', SAVE)
'''

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)
header=md('''# Сегментация воды и оценка затопления — Sen1Floods11

**Курсовая:** «Терең оқыту негізінде спутниктік суреттерден су басқан аумақтарды анықтау және сегментациялау».

U-Net + ResNet-34, VV/VH, BCE + Dice. Метрики воды отделены от оценки затопления.
Исходные данные: [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11), Bonafilia et al., CVPR Workshops 2020.

Выберите **Среда выполнения → Сменить среду выполнения → T4 GPU** (если доступен).
Запускайте ячейки сверху вниз. Подключение Drive требует вашего разрешения: код сохраняет результаты в `MyDrive/FloodSegmentation`.
Google может прервать сессию; `last.pt` позволяет продолжить. После перезапуска повторите настройку, подключение Drive и загрузку данных, затем ячейку обучения.
Данные хранятся в `/content`, результаты — на Drive. Публичная публикация не требуется.
''')
common=[header,md('## 1. Код проекта и зависимости\nPyTorch из Colab сохраняется. Устанавливаются недостающие библиотеки.'),code(bootstrap),md('## 2. Постоянное хранилище'),code(storage),md('## 3. Проверки реализации'),code(tests),md('## 4. Загрузка, аудит, train-only нормализация'),code(data),code(preview)]
allcells=common+[md('## 5. Проверка переобучения на нескольких патчах'),code(smoke),md('## 6. Основное обучение\nДо 100 эпох; early stopping patience=15. Не меняйте конфигурацию существующего запуска при resume.'),code(training),md('## 7. Независимое тестирование\nПорог выбирается только на validation. Bolivia — отдельный holdout.'),code(evaluation),md('## 8. Пороговый baseline Otsu'),code(baseline),md('## 9. Обычная U-Net\nОтдельный полноценный эксперимент; может потребовать следующей GPU-сессии.'),code(plain),md('## 10. Карты предполагаемого затопления\nJRC применяется только после сегментации. Возможны сезонная вода и ошибки карты постоянных водоёмов.'),code(maps),md('## 11. Итоговая таблица и рисунки'),code(summary)]
notebooks={'FloodSegmentation_Colab.ipynb':allcells,'01_data_audit.ipynb':common,'02_train_evaluate.ipynb':common+[code(smoke),code(training),code(evaluation),code(baseline),code(plain),code(summary)],'03_maps.ipynb':common+[code(maps),code(summary)]}
for name,cells in notebooks.items():
    book=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':name},'accelerator':'GPU' if name!='01_data_audit.ipynb' else 'None'})
    nbf.validate(book)
    for cell in cells:
        if cell.cell_type=='code': compile(cell.source,name,'exec')
    dest=ROOT/'notebooks'/name; dest.parent.mkdir(exist_ok=True); nbf.write(book,dest)
    print(dest.name, len(cells), 'cells')
