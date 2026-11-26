import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import torch
from flood.improve import (boundary_map, lovasz_hinge_per_image, boundary_lovasz_loss, train_v2,
                           evaluate_v2, predict_v2, flatten, mean_std, patched_loss)
import flood.train as T


def make_tile(path_img, path_lbl, seed, offset):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:512, :512]
    cy, cx, r = rng.integers(100, 400, 2).tolist() + [int(rng.integers(20, 120))]
    lab = (((yy - cy) ** 2 + (xx - cx) ** 2) < r * r).astype('float32')
    lab[:40, :40] = -1
    img = np.stack([np.where(lab == 1, -20., -8.), np.where(lab == 1, -26., -14.)]).astype('float32')
    img += rng.normal(0, 2, img.shape).astype('float32')
    for p, a in [(path_img, img), (path_lbl, lab[None])]:
        with rasterio.open(p, 'w', driver='GTiff', height=512, width=512, count=len(a), dtype='float32',
                           crs='EPSG:4326', transform=from_origin(10 + offset, 10, .0001, .0001)) as d:
            d.write(a)


class ImproveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory(); root = Path(cls.tmp.name); rows = []
        k = 0
        for split, n in [('train', 3), ('valid', 2), ('test', 2), ('bolivia', 1)]:
            for i in range(n):
                chip = ('Bolivia' if split == 'bolivia' else 'Spain') + f'_{k}'
                img, lbl = root / f'{chip}_img.tif', root / f'{chip}_lbl.tif'
                make_tile(img, lbl, k, k); k += 1
                rows.append(dict(chip_id=chip, event=chip.split('_')[0], split=split, image=str(img), label=str(lbl)))
        cls.df = pd.DataFrame(rows); cls.root = root
        cls.norm = {'mean': [-10., -16.], 'std': [5., 5.]}

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_boundary_map_marks_edges_only(self):
        y = torch.zeros(1, 1, 20, 20); y[..., 5:15, 5:15] = 1
        v = torch.ones_like(y, dtype=torch.bool)
        b = boundary_map(y, v, 1)
        self.assertEqual(b[0, 0, 10, 10].item(), 0); self.assertEqual(b[0, 0, 0, 0].item(), 0)
        self.assertEqual(b[0, 0, 5, 10].item(), 1); self.assertEqual(b[0, 0, 4, 10].item(), 1)

    def test_lovasz_ignores_invalid_and_rewards_correct(self):
        y = torch.zeros(2, 1, 8, 8); y[0, :, :4] = 1
        v = torch.ones_like(y, dtype=torch.bool); v[1] = False
        good = (y * 2 - 1) * 5; bad = -good
        lg, lb = lovasz_hinge_per_image(good, y, v), lovasz_hinge_per_image(bad, y, v)
        self.assertLess(lg.item(), lb.item())
        bad_invalid = good.clone(); bad_invalid[1] = 100  # garbage in invalid image must not matter
        self.assertAlmostEqual(lovasz_hinge_per_image(bad_invalid, y, v).item(), lg.item(), places=6)
        self.assertIsNone(lovasz_hinge_per_image(good, y, torch.zeros_like(v)))

    def test_loss_ignores_nan_outside_valid_and_backprops(self):
        z = torch.randn(2, 1, 16, 16, requires_grad=True); y = (torch.rand(2, 1, 16, 16) > .5).float()
        v = torch.ones_like(y, dtype=torch.bool); v[..., :3, :] = False
        y2 = y.clone(); y2[~v] = float('nan')
        l = boundary_lovasz_loss(z, torch.nan_to_num(y2), v)
        self.assertTrue(torch.isfinite(l)); l.backward(); self.assertTrue(torch.isfinite(z.grad).all())

    def test_patch_restores_loss(self):
        orig = T.masked_loss
        with patched_loss({'loss': 'boundary_lovasz'}):
            self.assertIsNot(T.masked_loss, orig)
        self.assertIs(T.masked_loss, orig)

    def test_tta_equivariance_and_full_mode(self):
        from flood.models import build_model
        m = build_model('attention_resnet34', False).eval()
        x, y, v = np.random.randn(2, 512, 512).astype('float32'), None, np.ones((512, 512), bool)
        n = {'mean': [0, 0], 'std': [1, 1]}
        for mode in ('patch', 'full'):
            p = predict_v2(m, x, v, n, torch.device('cpu'), mode, tta=True)
            self.assertEqual(p.shape, (512, 512)); self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_end_to_end_train_resume_evaluate(self):
        out = self.root / 'run'
        cfg = {'model': 'attention_resnet34', 'pretrained': False, 'seed': 7, 'epochs': 1, 'workers': 0,
               'batch_size': 4, 'loss': 'boundary_lovasz', 'boundary_weight': 2.0}
        ck = train_v2(self.df, self.norm, out, cfg, allow_cpu=True)
        train_v2(self.df, self.norm, out, {**cfg, 'epochs': 2}, allow_cpu=True)
        state = torch.load(out / 'last.pt', weights_only=False)
        self.assertEqual(state['epoch'], 1); self.assertEqual(state['config']['loss'], 'boundary_lovasz')
        self.assertIs(T.masked_loss.__name__, 'masked_loss')
        s = evaluate_v2(self.df, ck, self.root / 'ev', mode='full', tta=True)
        for split in ('valid', 'test', 'bolivia'):
            self.assertIn('edge_recall', s[split]); self.assertTrue(0 <= s[split]['iou'] <= 1)
        rows = flatten('attention', 7, 'full+tta', s) + flatten('attention', 8, 'full+tta', s)
        t = mean_std(pd.DataFrame(rows)); self.assertEqual(int(t.n_seeds.iloc[0]), 2)
        self.assertTrue((self.root / 'ev' / 'per_tile.csv').exists())
        s2 = evaluate_v2(self.df, ck, self.root / 'ev', mode='full', tta=True)  # cached
        self.assertEqual(s, s2)


if __name__ == '__main__':
    unittest.main()
