import unittest
import numpy as np
import torch
from flood.core import counts, metrics, masked_loss, assemble, validate_splits

class CoreTests(unittest.TestCase):
    def test_metrics_and_invalid(self):
        y = np.array([1, 0, 1, 0]); p = np.array([1, 1, 0, 1]); v = np.array([1, 1, 1, 0], bool)
        c = counts(p, y, v)
        self.assertEqual(c.tolist(), [1, 1, 1, 0])
        self.assertAlmostEqual(metrics(c)['iou'], 1/3)
        self.assertAlmostEqual(metrics(c)['f1'], .5)

    def test_loss_ignores_invalid(self):
        y = torch.tensor([[[[1., 0.]]]])
        v = torch.tensor([[[[True, False]]]])
        a = torch.tensor([[[[0., -100.]]]], requires_grad=True)
        b = torch.tensor([[[[0., 100.]]]])
        self.assertEqual(masked_loss(a,y,v).item(), masked_loss(b,y,v).item())
        masked_loss(a,y,v).backward()
        self.assertEqual(a.grad[0,0,0,1].item(),0)
        self.assertIsNone(masked_loss(a,y,torch.zeros_like(v)))

    def test_empty_target_penalized(self):
        y = torch.zeros(1,1,4,4); v = torch.ones_like(y,dtype=torch.bool)
        self.assertGreater(masked_loss(y+5,y,v),masked_loss(y-5,y,v))

    def test_assemble(self):
        tiles = [np.full((2,2),i) for i in range(4)]
        out = assemble(tiles,4,2)
        np.testing.assert_array_equal(out,[[0,0,1,1],[0,0,1,1],[2,2,3,3],[2,2,3,3]])

    def test_split_leakage(self):
        with self.assertRaises(ValueError):
            validate_splits([{'chip_id':'a','split':'train','event':'Spain'}, {'chip_id':'a','split':'test','event':'Spain'}])
        with self.assertRaises(ValueError):
            validate_splits([{'chip_id':'a','split':'train','event':'Bolivia'}])

if __name__ == '__main__': unittest.main()
