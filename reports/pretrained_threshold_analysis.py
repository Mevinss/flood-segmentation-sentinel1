"""Run in the project Colab after mounting Drive and preparing the data."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import maximum_filter, minimum_filter
from flood.core import counts, metrics
from flood.data import read_pair, read_image
from flood.evaluate import load_model, predict

save = Path('/content/drive/MyDrive/FloodSegmentation')
dest = save / 'pretrained_threshold_analysis'
dest.mkdir(exist_ok=True)
reference = save / 'evaluation/resnet34'
original = json.loads((reference / 'threshold.json').read_text())
manifest = pd.read_csv('/content/flood_project/data/manifest.csv')
model, normalization, device = load_model(save / 'runs/resnet34_seed42/best.pt')
grid = np.array([r['threshold'] for r in original['curve']])
totals = np.zeros((len(grid), 4), dtype=np.int64)
for row in manifest.query('split=="valid"').to_dict('records'):
    x, target, valid = read_pair(row)
    _, image_valid = read_image(row)
    prob = predict(model, x, image_valid, normalization, device)
    for i, t in enumerate(grid):
        totals[i] += counts(prob >= t, target, valid)
curve = pd.DataFrame([dict(threshold=t, **metrics(c)) for t, c in zip(grid, totals)])
curve.to_csv(dest / 'validation.csv', index=False)
best = float(curve.assign(distance=(curve.threshold-.5).abs()).sort_values(
    ['iou','distance','threshold'], ascending=[False,True,True]).iloc[0].threshold)
print('VALIDATION', curve[['threshold','precision','recall','iou']].to_string(index=False))
print('BEST', best, 'original', original['threshold'])
fig, ax = plt.subplots(figsize=(7,4))
for col in ['precision','recall','iou']:
    ax.plot(grid, curve[col], marker='.', label=col)
ax.axvline(best, linestyle=':', color='black', label=f'Порог {best:.2f}')
ax.set(xlabel='Порог вероятности воды', ylabel='Значение метрики', title='ResNet-34 с ImageNet · валидация')
ax.legend()
fig.tight_layout()
fig.savefig(dest/'validation.png',dpi=170)
plt.close(fig)

records = []
colors = ['#e7e7e7','#2879a2','#dd341e','#e1a639','#515151']
for row in manifest.query('split=="bolivia"').to_dict('records'):
    chip = row['chip_id']
    x, y, valid = read_pair(row)
    with rasterio.open(reference/'probabilities'/(chip+'.tif')) as src, rasterio.open(row['image']) as ref:
        assert (src.shape,src.transform,src.crs)==(ref.shape,ref.transform,ref.crs)
        prob = src.read(1,masked=True)
        assert not np.ma.getmaskarray(prob)[valid].any() and np.isfinite(prob.data[valid]).all()
    water = (y==1)&valid
    land = valid & ~water
    pred = prob.data >= best
    fp = land & pred
    known = minimum_filter(valid.astype('uint8'), size=5, mode='constant',cval=0).astype(bool)
    near = maximum_filter(water.astype('uint8'),size=5).astype(bool)
    zones = {'near':land & known & near,'far':land & known & ~near,'unknown':land & ~known}
    rec = dict(chip=chip, **metrics(counts(pred,y,valid)))
    for name, zone in zones.items():
        rec['land_'+name] = int(zone.sum())
        rec['fp_'+name] = int((zone & fp).sum())
    records.append(rec)
    error = np.zeros(valid.shape,'uint8')
    error[water & pred]=1
    error[fp]=2
    error[water & ~pred]=3
    error[~valid]=4
    fig, axes = plt.subplots(1,4,figsize=(14,4))
    for ax, band, label in zip(axes[:2], x, ['VV','VH']):
        lo,hi=np.percentile(band[valid],[2,98])
        ax.imshow(band,cmap='gray',vmin=lo,vmax=hi)
        ax.set_title(label+' (дБ)')
    axes[2].imshow(np.where(valid,water,2),cmap=ListedColormap([colors[0],colors[1],colors[4]]),vmin=0,vmax=2,interpolation='nearest')
    axes[2].set_title('Ручная разметка')
    axes[3].imshow(error,cmap=ListedColormap(colors),vmin=0,vmax=4,interpolation='nearest')
    axes[3].set_title('Ошибки модели')
    for ax in axes: ax.axis('off')
    fig.suptitle(f'{chip} · порог {best:.2f} · ложные срабатывания {int(fp.sum()):,} пикс.')
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,['Суша (TN)','Вода (TP)','Суша принята за воду (FP)','Пропуск воды (FN)','Нет разметки'])],loc='lower center',ncol=5)
    fig.tight_layout(rect=(0,.10,1,.94))
    fig.savefig(dest/(chip+'.png'),dpi=135)
    plt.close(fig)
chips = pd.DataFrame(records).sort_values('fp',ascending=False)
chips.to_csv(dest/'bolivia_fp.csv',index=False)
assert len(chips)==15 and (chips.fp==chips[['fp_near','fp_far','fp_unknown']].sum(axis=1)).all()
if best==original['threshold']:
    baseline=json.loads((reference/'metrics.json').read_text())['bolivia']
    assert all(chips[k].sum()==baseline[k] for k in ['tp','fp','fn','tn'])
print('FP MAPS COMPLETE', len(chips), int(chips.fp.sum()))
from google.colab import files
files.download(shutil.make_archive('/content/pretrained_threshold_analysis','zip',dest))
