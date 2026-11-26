# Plain U-Net: same data, loss, seed and selection rule as ResNet-34
from pathlib import Path
import json, pandas as pd, torch
from IPython.display import display, Image
from flood.data import prepare, audit
from flood.train import train, DEFAULT
from flood.evaluate import evaluate, learning_curves
assert Path('/content/drive/MyDrive').exists(), 'Mount Google Drive first'
SAVE = Path('/content/drive/MyDrive/FloodSegmentation')
assert torch.cuda.is_available(), 'GPU required'
df = prepare(WORK / 'data', include_jrc=True)
assert len(df) == 446
norm, stats = audit(df, SAVE / 'audit_plain')
reference_norm = json.loads((SAVE / 'runs/resnet34_seed42/normalization.json').read_text())
assert norm == reference_norm, 'Normalization mismatch with ResNet experiment'
plain_run = SAVE / 'runs/plain_seed42'
plain_checkpoint = train(df, norm, plain_run, {**DEFAULT, 'model':'plain', 'pretrained':False}, resume=True)
plain_metrics = evaluate(df, SAVE / 'evaluation/plain', plain_checkpoint)
learning_curves(plain_run)
print('PLAIN U-NET RESULTS:', json.dumps(plain_metrics, indent=2))
display(Image(filename=str(plain_run / 'learning_curves.png')))
rows = []
for name in ['otsu', 'plain', 'resnet34']:
    result = json.loads((SAVE / 'evaluation' / name / 'metrics.json').read_text())
    for split, values in result.items():
        rows.append({'model': name, 'split': split, **values})
comparison = pd.DataFrame(rows)
comparison.to_csv(SAVE / 'comparison.csv', index=False)
display(comparison[['model','split','iou','f1','precision','recall']])
for name in ['plain','resnet34']:
    h = pd.read_csv(SAVE / 'runs' / (name+'_seed42') / 'history.csv')
    print(name, 'BEST VALIDATION:', h.loc[h.valid_iou.idxmax()].to_dict())
print('COMPARISON COMPLETE:', SAVE / 'comparison.csv')
