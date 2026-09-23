#!/usr/bin/env python
from pathlib import Path
import shutil, numpy as np, pandas as pd
root = Path("runs/smoke")
if root.exists(): shutil.rmtree(root)
(root/"tokens").mkdir(parents=True)
rng = np.random.default_rng(123); rows=[]
for i in range(5000):
    slp=float(np.clip(rng.normal(-2.8,1.8),-12,-.01)); sp=float(np.exp(slp))
    tp=float(np.clip(sp+rng.normal(0,.03),1e-8,1)); tlp=float(np.log(tp)); a=tlp-slp
    rows.append({"problem_local_index":i//500,"problem_source_index":i//500,"problem_id":str(i//500),"rollout_id":(i//100)%5,
    "token_position":i,"token_id":100+i%100,"token_text":"x","student_prob":sp,"student_logprob":slp,"student_entropy":float(rng.uniform(.1,4)),
    "student_rank":int(rng.integers(1,100)),"student_top1_prob":float(rng.uniform(sp,1)),"teacher_prob":tp,"teacher_logprob":tlp,
    "teacher_entropy":float(rng.uniform(.1,4)),"teacher_rank":int(rng.integers(1,100)),"teacher_top1_prob":float(rng.uniform(tp,1)),
    "prob_diff_signed":tp-sp,"prob_diff_abs":abs(tp-sp),"logratio_signed":a,"logratio_abs":abs(a),
    "grad_proxy_signed":a*(1-sp),"grad_proxy_abs":abs(a)*(1-sp)})
pd.DataFrame(rows).to_parquet(root/"tokens"/"synthetic.parquet",index=False)
Path("configs/analysis_smoke.yaml").write_text('''input_dir: runs/smoke\noutput_dir: runs/smoke/analysis\nbinning:\n  mode: fixed\n  logp_min: -12\n  logp_max: 0\n  bin_width: 0.5\nstatistics:\n  central: median\n  show_iqr: true\nplot:\n  format: pdf\n  dpi: 120\noutput:\n  write_consolidated_parquet: true\n''')
print("Run: PYTHONPATH=src python stage2_analyze.py --config configs/analysis_smoke.yaml")
