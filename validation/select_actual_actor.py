"""Derive the exact retained actor batch from the frozen natural-correction replay."""
from pathlib import Path
import json,sys
import numpy as np,torch
R=Path(__file__).resolve().parent;E=R/'evidence';sys.path.insert(0,str(R.parent/'verl'))
record=json.loads((E/'natural_correction_first.json').read_text());audit=Path(record['hook_output']).parent
actor_path=sorted(audit.glob('*trainer_actor_batch.pt'))[0];actual=torch.load(actor_path,weights_only=False)
source=E/'replay_trainer_natural/actor_input.pt';frozen=torch.load(source,weights_only=False);b=frozen['batch']
lookup={str(t):i for i,t in enumerate(b.non_tensor_batch['trajectory_id'])};ix=[lookup[str(t)] for t in actual.non_tensor_batch['trajectory_id']];selected=b[ix]
for key in ['prompts','responses','input_ids','position_ids','attention_mask','response_mask']:assert torch.equal(selected.batch[key],actual.batch[key]),key
selected.batch['old_log_probs']=actual.batch['old_log_probs'].clone()
extras={k:np.asarray(v)[ix] for k,v in frozen['extras'].items()}
out=E/'replay_trainer_natural/actor_actual_input.pt';assert not out.exists()
torch.save(dict(batch=selected,raw=frozen['raw'][ix],extras=extras,scores={k:v[ix] for k,v in frozen['scores'].items()}),out)
(E/'replay_trainer_natural/actor_selection.json').write_text(json.dumps(dict(status='PASS',source=str(source),actual_actor_batch=str(actor_path),candidate_rows=ix,rows=len(actual),input_keys_exact=True,uses_actual_old_log_probs=True,scope='Exact actually retained first actor batch at version0; no additional trajectory selection'),indent=2))
print(out)
