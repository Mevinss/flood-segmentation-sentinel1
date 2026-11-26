"""Run after attention training in the project Colab runtime."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from flood.core import counts, metrics
from flood.data import read_pair, read_image
from flood.evaluate import load_model, predict

save = Path('/content/drive/MyDrive/FloodSegmentation')
run = save / 'runs/attention_resnet34_seed42'
out_dir = save / 'evaluation/attention_matched_recall'
out_dir.mkdir(exist_ok=True, parents=True)
baseline_dir = save / 'evaluation/resnet34'
reference = pd.read_csv(save / 'pretrained_threshold_analysis/validation.csv')
threshold_ref = json.loads((baseline_dir / 'threshold.json').read_text())['threshold']
target = reference.loc[np.isclose(reference.threshold, threshold_ref), 'recall'].item()
grid = np.round(np.arange(.001, 1, .001), 3)
model, norm, device = load_model(run / 'best.pt')
frame = pd.read_csv('/content/flood_project/data/manifest.csv')

def scored(row):
    image, y, valid = read_pair(row)
    _, image_valid = read_image(row)
    p = predict(model, image, image_valid, norm, device)[valid]
    assert np.isfinite(p).all()
    return p, y[valid].astype(bool), valid

def predicted_positive(scores):
    return len(scores) - np.searchsorted(np.sort(scores), grid, side='left')

tp = np.zeros(len(grid), dtype='int64')
fp = np.zeros_like(tp)
n_water = n_land = 0
for row in frame.query('split=="valid"').to_dict('records'):
    p, y, _ = scored(row)
    tp += predicted_positive(p[y])
    fp += predicted_positive(p[~y])
    n_water += y.sum()
    n_land += (~y).sum()
curve = pd.DataFrame([dict(threshold=t, **metrics([a,b,n_water-a,n_land-b])) for t,a,b in zip(grid,tp,fp)])
curve.to_csv(out_dir/'validation.csv', index=False)
chosen = curve.assign(gap=(curve.recall-target).abs()).sort_values(['gap','fp','threshold']).iloc[0]
t = float(chosen.threshold)
print('VALIDATION OPERATING POINTS', threshold_ref, target, t, chosen.recall)

records = []
water_scores, land_scores = [], []
for row in frame.query('split=="bolivia"').to_dict('records'):
    chip = row['chip_id']
    p, y, valid = scored(row)
    with rasterio.open(baseline_dir/'probabilities'/(chip+'.tif')) as src, rasterio.open(row['image']) as ref:
        assert (src.crs,src.transform,src.shape)==(ref.crs,ref.transform,ref.shape)
        ref_p = src.read(1, masked=True)[valid]
        assert not np.ma.getmaskarray(ref_p).any()
    water_scores.append(p[y])
    land_scores.append(p[~y])
    for name, prediction in [('baseline',ref_p.data>=threshold_ref),('attention',p>=t)]:
        records.append(dict(chip=chip, model=name, **metrics(counts(prediction,y,np.ones_like(y)))))
chips = pd.DataFrame(records)
chips.to_csv(out_dir/'paired_chips.csv',index=False)
summary = {name:metrics(rows[['tp','fp','fn','tn']].sum().to_numpy()) for name,rows in chips.groupby('model')}
expected = json.loads((baseline_dir/'metrics.json').read_text())['bolivia']
assert all(summary['baseline'][k]==expected[k] for k in ['tp','fp','fn','tn'])
assert chips.chip.nunique() == 15

# Descriptive point on the Bolivia ROC at exactly baseline TP. No deployment threshold is selected here.
water = np.sort(np.concatenate(water_scores))
land = np.sort(np.concatenate(land_scores))
k = summary['baseline']['tp']
cut = water[-k]
above_tp = len(water) - np.searchsorted(water,cut,'right')
equal_tp = np.searchsorted(water,cut,'right') - np.searchsorted(water,cut,'left')
above_fp = len(land) - np.searchsorted(land,cut,'right')
equal_fp = np.searchsorted(land,cut,'right') - np.searchsorted(land,cut,'left')
interp_fp = float(above_fp+(k-above_tp)/equal_tp*equal_fp)
history = pd.read_csv(run/'history.csv')
result = dict(bolivia=summary, tolerance_pp=2, threshold_base=threshold_ref,
              validation_target=target, threshold_attention=t, validation_attention=float(chosen.recall),
              matched_tp=k, diagnostic_fp_attention=interp_fp, epochs=len(history),
              best_epoch=int(history.loc[history.valid_iou.idxmax(),'epoch']),best_valid_iou=float(history.valid_iou.max()))
(out_dir/'results.json').write_text(json.dumps(result,indent=2))
print('ATTENTION COMPARISON COMPLETE',json.dumps(result),flush=True)
for filename in ['history.csv','config.json','learning_curves.png','models.py']:
    shutil.copy2(run/filename,out_dir/filename)
from google.colab import files
files.download(shutil.make_archive('/content/attention_matched_recall','zip',out_dir))
