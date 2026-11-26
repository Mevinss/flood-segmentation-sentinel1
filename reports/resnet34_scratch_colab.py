from pathlib import Path
import json, pandas as pd, torch
from flood.data import prepare, audit
from flood.train import train
from flood.evaluate import evaluate, learning_curves
ref = SAVE/'runs/resnet34_seed42'
run = SAVE/'runs/resnet34_scratch_seed42'
out = SAVE/'evaluation/resnet34_scratch'
old = SAVE/'evaluation/resnet34'
config = json.loads((ref/'config.json').read_text())
assert config['pretrained'] and config['model']=='resnet34'
config['pretrained'] = False
df = prepare(WORK/'data', include_jrc=False)
norm, _ = audit(df, WORK/'audit_scratch')
assert norm == json.loads((ref/'normalization.json').read_text())
ref_state = torch.load(ref/'best.pt',weights_only=False,map_location='cpu')
assert ref_state['split_ids'] == {s:sorted(df[df.split==s].chip_id) for s in df.split.unique()}
del ref_state
checkpoint = train(df, norm, run, config, resume=True)
metrics_scratch = evaluate(df, out, checkpoint)
learning_curves(run)
print('SCRATCH METRICS',json.dumps(metrics_scratch))
