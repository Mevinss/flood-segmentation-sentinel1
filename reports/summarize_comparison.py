"""Export the observed Colab comparison; no training or metric estimation."""
import csv
import re
from pathlib import Path

root = Path(__file__).resolve().parent
evidence = (root / 'colab_comparison_evidence.txt').read_text(encoding='utf-8')
pattern = r'row "\d+ (otsu|plain|resnet34) (bolivia|test) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"'
rows = [dict(zip(['model','split','iou','f1','precision','recall'], m)) for m in re.findall(pattern, evidence)]
assert len(rows) == 6 and len({(r['model'],r['split']) for r in rows}) == 6
with (root / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
epochs = re.findall(r'Epoch (\d+): train loss=([\d.]+), valid IoU=([\d.]+), ([\d.]+)s', evidence)
assert len(epochs) == 62
with (root / 'plain_training_log.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f); w.writerow(['epoch','train_loss','valid_iou','seconds']); w.writerows(epochs)
assert max(epochs, key=lambda x: float(x[2]))[0] == '47'
print('Verified: 6 comparison rows, 62 epochs, best epoch 47. CSV values are rounded Colab table outputs.')
