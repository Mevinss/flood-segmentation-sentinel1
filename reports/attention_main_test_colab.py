"""Colab: жоба кітапханалары орнатылып, Google Drive қосылған соң орындаңыз."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import importlib
import json
import numpy as np
import pandas as pd
import torch
import flood.models
import flood.evaluate
from flood.core import counts, metrics
from flood.data import BASE, HAND, download, read_pair, read_image

root = Path('/content/drive/MyDrive/FloodSegmentation')
run_dir = root / 'runs/attention_resnet34_seed42'
threshold = 0.473
previous = json.loads((root / 'evaluation/attention_matched_recall/results.json').read_text())
assert previous['threshold_attention'] == threshold
Path(flood.models.__file__).write_text((run_dir / 'models.py').read_text())
importlib.reload(flood.models)
importlib.reload(flood.evaluate)
checkpoint = run_dir / 'best.pt'
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert saved['config']['model'] == 'attention_resnet34' and saved['epoch'] + 1 == previous['best_epoch']
expected_ids = saved['split_ids']['test']
del saved
model, norm, device = flood.evaluate.load_model(checkpoint)
data_dir = Path('/content/flood_project/data')
split_file = data_dir / 'splits/flood_test_data.csv'
download(BASE + 'splits/flood_handlabeled/' + split_file.name, split_file)
with split_file.open() as source:
    pairs = [r for r in csv.reader(source) if r]
def fetch(pair):
    image, label = pair
    row = {'chip_id': image.removesuffix('_S1Hand.tif')}
    for key, folder, name in [('image', 'S1Hand', image), ('label', 'LabelHand', label)]:
        dest = data_dir / folder / name
        download(BASE + HAND + folder + '/' + name, dest)
        row[key] = dest
    return row
with ThreadPoolExecutor(max_workers=6) as pool:
    rows = list(pool.map(fetch, pairs))
assert sorted(row['chip_id'] for row in rows) == sorted(expected_ids)
records = []
for row in rows:
    x, y, valid = read_pair(row)
    _, image_valid = read_image(row)
    p = flood.evaluate.predict(model, x, image_valid, norm, device)
    assert np.isfinite(p[valid]).all()
    records.append(dict(chip=row['chip_id'], **metrics(counts(p >= threshold, y, valid))))
table = pd.DataFrame(records)
result = metrics(table[['tp','fp','fn','tn']].sum().to_numpy())
base = json.loads((root / 'evaluation/resnet34/metrics.json').read_text())['test']
for a,b in [('tp','fn'),('fp','tn')]:
    assert result[a]+result[b] == base[a]+base[b]
out = root / 'evaluation/attention_main_test'
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out / 'per_tile.csv', index=False)
result.update(threshold=threshold, tiles=len(rows), epoch=previous['best_epoch'])
(out / 'results.json').write_text(json.dumps(result, indent=2))
print('Негізгі тест аяқталды', json.dumps(result))
