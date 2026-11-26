"""Compare two validation-thresholded probability rasters on common labelled water."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import minimum_filter, maximum_filter

from flood.data import read_pair


def compare_bolivia(frame, pretrained_dir, scratch_dir, destination):
    destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)
    directories = [Path(pretrained_dir), Path(scratch_dir)]
    thresholds = [json.loads((d / 'threshold.json').read_text())['threshold'] for d in directories]
    palette = ['#e4e4e4', '#28749c', '#e77700', '#813d98', '#c72e3e', '#505050']
    labels = ['Суша', 'Воду нашли обе', 'Пропуск ImageNet', 'Пропуск с нуля', 'Пропуск обеих', 'Нет разметки']
    records = []
    for row in frame[frame.split == 'bolivia'].sort_values('chip_id').to_dict('records'):
        chip = row['chip_id']
        radar, mask, valid = read_pair(row)
        water = valid & (mask == 1)
        predictions = []
        with rasterio.open(row['image']) as reference:
            for directory, threshold in zip(directories, thresholds):
                with rasterio.open(directory / 'probabilities' / (chip + '.tif')) as src:
                    assert (src.shape, src.crs, src.transform) == (reference.shape, reference.crs, reference.transform)
                    probability = src.read(1, masked=True)
                    assert not np.ma.getmaskarray(probability)[valid].any()
                    assert np.isfinite(probability.data[valid]).all()
                    predictions.append(probability.data >= threshold)
        p, s = predictions
        # Distances on the annotation grid. Exclude the 2-pixel rim near unknowns
        # or tile borders from boundary/interior comparisons.
        known = minimum_filter(valid.astype('uint8'), size=5, mode='constant', cval=0).astype(bool)
        near_land = maximum_filter((valid & ~water).astype('uint8'), size=5).astype(bool)
        zones = {'all': water, 'edge_2px': water & known & near_land,
                 'interior': water & known & ~near_land, 'unknown_nearby': water & ~known}
        for zone, sample in zones.items():
            records.append(dict(chip=chip, zone=zone, water=int(sample.sum()),
                                missed_pretrained=int((sample & ~p).sum()),
                                missed_scratch=int((sample & ~s).sum()),
                                missed_both=int((sample & ~p & ~s).sum())))
        classes = np.zeros(water.shape, dtype='uint8')
        classes[water & p & s] = 1
        classes[water & ~p & s] = 2
        classes[water & p & ~s] = 3
        classes[water & ~p & ~s] = 4
        classes[~valid] = 5
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, band, name in zip(axes[:2], radar, ['VV', 'VH']):
            lo, hi = np.percentile(band[valid], [2, 98]) if valid.any() else (-30, 0)
            ax.imshow(band, cmap='gray', vmin=lo, vmax=hi)
            ax.set_title(name + ' (дБ)')
        axes[2].imshow(np.where(valid, water, 2), cmap=ListedColormap([palette[0], palette[1], palette[5]]), vmin=0, vmax=2)
        axes[2].set_title('Ручная разметка воды')
        axes[3].imshow(classes, cmap=ListedColormap(palette), vmin=0, vmax=5, interpolation='nearest')
        axes[3].set_title('Совпадения и пропуски')
        for ax in axes:
            ax.axis('off')
        fig.suptitle(chip)
        fig.legend(handles=[Patch(color=c, label=t) for c, t in zip(palette, labels)],
                   loc='lower center', ncol=6, fontsize=9)
        fig.tight_layout(rect=(0, .09, 1, .95))
        fig.savefig(destination / (chip + '.png'), dpi=140)
        plt.close(fig)
    table = pd.DataFrame(records)
    table.to_csv(destination / 'paired_by_chip.csv', index=False)
    sums = table.groupby('zone').sum(numeric_only=True)
    sums.to_csv(destination / 'paired_by_zone.csv')
    for name, directory in zip(['pretrained', 'scratch'], directories):
        metrics = json.loads((directory / 'metrics.json').read_text())['bolivia']
        assert sums.loc['all', 'missed_' + name] == metrics['fn']
        assert sums.loc['all', 'water'] == metrics['tp'] + metrics['fn']
    return sums
