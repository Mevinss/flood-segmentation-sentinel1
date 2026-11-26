"""Improvements for the flood-segmentation coursework.

1) Boundary errors   -> boundary-weighted BCE + Lovasz loss, full-tile (512x512) inference, test-time augmentation (TTA).
2) Low-water tiles   -> per-image Lovasz hinge: every tile contributes equally, not only large water bodies.
3) Single seed       -> helpers to train/evaluate several seeds and report mean +- std.

All selection decisions (threshold, inference mode, architecture) use the VALIDATION split only.
"""
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, minimum_filter

import flood.train as T
from flood.core import counts, metrics, write_json, masked_loss
from flood.data import read_pair, read_image, normalize
from flood.models import build_model

THRESHOLDS = np.round(np.arange(.1, .901, .05), 2)


# --------------------------------------------------------------------------- loss
def boundary_map(target, valid, radius=2):
    """1 where a labelled pixel lies within `radius` px of the water/land edge (morphological gradient)."""
    y = torch.where(valid, target, torch.zeros_like(target))
    k = 2 * radius + 1
    dil = F.max_pool2d(y, k, stride=1, padding=radius)
    ero = -F.max_pool2d(-y, k, stride=1, padding=radius)
    return ((dil - ero) > 0).float() * valid.float()


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jac = 1. - inter / union
    if len(gt_sorted) > 1:
        jac[1:] = jac[1:] - jac[:-1]
    return jac


def lovasz_hinge_per_image(logits, target, valid):
    """Lovasz hinge (Berman et al., 2018) computed per image on valid pixels only, then averaged.
    Images without labelled pixels are skipped."""
    losses = []
    for z, y, v in zip(logits, target, valid):
        z = z[v].float(); y = y[v].float()
        if z.numel() == 0:
            continue
        signs = 2. * y - 1.
        errors = 1. - z * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(F.relu(errors_sorted), lovasz_grad(y[perm])))
    if not losses:
        return None
    return torch.stack(losses).mean()


def boundary_lovasz_loss(logits, target, valid, boundary_weight=2.0, radius=2):
    if not valid.any():
        return None
    w = 1. + boundary_weight * boundary_map(target.float(), valid, radius)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
    bce = (bce * w)[valid].sum() / w[valid].sum()
    lov = lovasz_hinge_per_image(logits, target, valid)
    return bce + (lov if lov is not None else 0.)


@contextmanager
def patched_loss(cfg):
    """flood.train.epoch() calls the module-level masked_loss; swap it for the duration of training."""
    original = T.masked_loss
    if cfg.get('loss', 'bce_dice') == 'boundary_lovasz':
        bw = cfg.get('boundary_weight', 2.0); r = cfg.get('boundary_radius', 2)
        T.masked_loss = lambda z, y, v: boundary_lovasz_loss(z, y, v, bw, r)
    try:
        yield
    finally:
        T.masked_loss = original


def train_v2(df, norm, out, config, resume=True, allow_cpu=False):
    """Same training loop, checkpoints and resume logic as flood.train.train; only the loss can change."""
    with patched_loss(config):
        return T.train(df, norm, out, config, resume=resume, allow_cpu=allow_cpu)


# --------------------------------------------------------------------------- inference
def load_model(checkpoint, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_model(state['config']['model'], False).to(device)
    model.load_state_dict(state['model']); model.eval()
    return model, state['normalization'], device, state


@torch.inference_mode()
def predict_v2(model, x, valid, norm, device, mode='patch', tta=False):
    """mode='patch': four 256x256 patches (original protocol); mode='full': whole 512x512 tile in one pass.
    tta=True averages the 8 rotations/flips (same transforms as training augmentation)."""
    z = torch.from_numpy(normalize(x, valid, norm).copy())[None].to(device)

    def run(a):
        if mode == 'full':
            return model(a).sigmoid()
        out = torch.zeros_like(a[:, :1])
        for top in (0, 256):
            for left in (0, 256):
                out[..., top:top + 256, left:left + 256] = model(a[..., top:top + 256, left:left + 256]).sigmoid()
        return out

    if not tta:
        return run(z)[0, 0].float().cpu().numpy()
    acc = torch.zeros_like(z[:, :1])
    for k in range(4):
        for flip in (False, True):
            a = torch.rot90(z, k, (-2, -1))
            if flip: a = torch.flip(a, (-1,))
            p = run(a)
            if flip: p = torch.flip(p, (-1,))
            acc += torch.rot90(p, -k, (-2, -1))
    return (acc / 8)[0, 0].float().cpu().numpy()


def zones(y, valid, radius=2):
    """Same zones as the Bolivia analysis in the report: water near land (edge) vs interior water."""
    water = valid & (y == 1)
    k = 2 * radius + 1
    known = minimum_filter(valid.astype('uint8'), size=k, mode='constant', cval=0).astype(bool)
    near_land = maximum_filter((valid & ~water).astype('uint8'), size=k).astype(bool)
    return water & known & near_land, water & known & ~near_land


def water_bin(share):
    return '<2%' if share < .02 else ('2–15%' if share < .15 else '>15%')


def evaluate_v2(df, checkpoint, out, mode='patch', tta=False, cache=True):
    """Threshold chosen on validation (max IoU, same grid as the report); then pooled + per-tile metrics."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if cache and (out / 'summary.json').exists():
        return json.loads((out / 'summary.json').read_text())
    model, norm, device, state = load_model(checkpoint)
    rows = df[df.split.isin(['valid', 'test', 'bolivia'])].to_dict('records')
    probs = {}
    for r in rows:
        x, y, v = read_pair(r); _, iv = read_image(r)
        probs[r['chip_id']] = (predict_v2(model, x, iv, norm, device, mode, tta), y, v)
    totals = np.zeros((len(THRESHOLDS), 4), dtype=np.int64)
    for r in rows:
        if r['split'] != 'valid': continue
        p, y, v = probs[r['chip_id']]
        for i, t in enumerate(THRESHOLDS): totals[i] += counts(p >= t, y, v)
    scores = [metrics(c)['iou'] for c in totals]
    best = sorted(range(len(THRESHOLDS)), key=lambda i: (-scores[i], abs(THRESHOLDS[i] - .5)))[0]
    threshold = float(THRESHOLDS[best])
    records = []
    for r in rows:
        p, y, v = probs[r['chip_id']]; pred = p >= threshold
        edge, interior = zones(y, v)
        water = int((v & (y == 1)).sum()); n = int(v.sum())
        records.append(dict(chip_id=r['chip_id'], event=r['event'], split=r['split'], water_px=water, valid_px=n,
                            water_bin=water_bin(water / n) if n else 'none',
                            edge_water=int(edge.sum()), edge_tp=int((edge & pred).sum()),
                            interior_water=int(interior.sum()), interior_tp=int((interior & pred).sum()),
                            **metrics(counts(pred, y, v))))
    table = pd.DataFrame(records); table.to_csv(out / 'per_tile.csv', index=False)
    summary = {'threshold': threshold, 'mode': mode, 'tta': tta, 'epoch': int(state['epoch']) + 1}
    for split, g in table.groupby('split'):
        s = metrics(g[['tp', 'fp', 'fn', 'tn']].sum().to_numpy())
        s['edge_recall'] = g.edge_tp.sum() / max(g.edge_water.sum(), 1)
        s['interior_recall'] = g.interior_tp.sum() / max(g.interior_water.sum(), 1)
        s['median_tile_iou'] = float(g.iou.dropna().median())
        for b, gb in g.groupby('water_bin'):
            s[f'iou_water_{b}'] = metrics(gb[['tp', 'fp', 'fn', 'tn']].sum().to_numpy())['iou']
            s[f'tiles_water_{b}'] = int(len(gb))
        summary[split] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in s.items()}
    write_json(out / 'summary.json', summary)
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    return summary


# --------------------------------------------------------------------------- reporting
def flatten(name, seed, variant, summary):
    rows = []
    for split in ('valid', 'test', 'bolivia'):
        rows.append({'model': name, 'seed': seed, 'inference': variant, 'split': split,
                     'threshold': summary['threshold'], **summary[split]})
    return rows


def mean_std(table, keys=('iou', 'f1', 'precision', 'recall', 'edge_recall', 'interior_recall', 'median_tile_iou')):
    keys = [k for k in keys if k in table]
    g = table.groupby(['model', 'inference', 'split'])[keys]
    m = g.mean(); s = g.std(ddof=1)
    out = m.copy().astype(object)
    for k in keys:
        out[k] = [f'{100 * a:.2f} ± {100 * b:.2f}' if np.isfinite(b) else f'{100 * a:.2f}' for a, b in zip(m[k], s[k])]
    out['n_seeds'] = g.size()
    return out.reset_index()
