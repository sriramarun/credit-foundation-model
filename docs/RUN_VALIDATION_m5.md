# Run Validation — M5 Calendar-Out-of-Time Program

A read-only audit checklist that back-traces the M5 run stage by stage and proves each artifact is
correct. Run it on the container. This is verification, not re-running. The headline result being
validated: the credit foundation model, trained only on data ≤ Dec-2022, beats the same-window
XGBoost baseline on genuinely unseen 2023–2024 defaults (full: ROC 0.8257 / AP 0.0113 vs
0.7913 / 0.0057).

## Setup (once)

```bash
cd /workspace/credit-foundation-model
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/.gcloud/credit-fm-sa.json
export ROOT=gs://sriram-credit-fm-data
export RUN=$ROOT/output/processed/fannie_mae/run_2000_2022
export ENC=$ROOT/output/encoded/fannie_mae/run_2000_2022
Invariants being proven
#	Stage	The one thing that makes it correct
0	Code	tests green, ruff clean at the run commit
1	Ingest	4% is a consistent hash; raw data complete
2	Prepare	loan-disjoint splits + every row ≤ Dec-2022
3	Tokenizer	fit on train only; no leakage columns; cal ≤ 2022Q4
4–5	Encode	loan counts match prepare; tokenizer lineage = v2
6	Pretrain	checkpoint lineage points to v2 tokenizer + capped train; val 0.2303
7–9	Fine-tune	train cutoffs ≤2021, test 2022–23, gated, loan-disjoint; FM beats baseline
0 · Code integrity
git log --oneline -12 | cat
ruff check src scripts tests && python -m pytest tests/ -q
Pass: tests green (77 passed), ruff clean.

1 · Ingest — panel_2000_2024.parquet
python - <<'PY'
import pyarrow.parquet as pq, gcsfs, pandas as pd
fs = gcsfs.GCSFileSystem()
p = "sriram-credit-fm-data/output/raw/fannie_mae/panel_2000_2024.parquet"
m = pq.ParquetFile(fs.open(p)).metadata
print("rows:", f"{m.num_rows:,}", "| columns:", m.num_columns)
df = pd.read_parquet(f"gs://{p}", columns=["loan_id","reporting_date","origination_date",
      "default_event","is_performing","current_interest_rate","current_loan_delinquency_status"])
print("reporting:", df.reporting_date.min(), "->", df.reporting_date.max())
print("origination:", df.origination_date.min(), "->", df.origination_date.max())
print("loans:", df.loan_id.nunique(), "| default rows:", int(df.default_event.fillna(False).sum()))
PY
Pass: ~125,027,505 rows; reporting 2000-01→2024-12; ~2,264,282 loans. current_loan_delinquency_status exists here (raw) — it must be absent from FM/baseline features later (leakage).

4% consistent-hash sampling:

python - <<'PY'
import pandas as pd
df = pd.read_parquet("gs://sriram-credit-fm-data/output/raw/fannie_mae/panel_2000_2024.parquet",
                     columns=["loan_id"])
h = pd.util.hash_pandas_object(df.loan_id.drop_duplicates(), index=False) % 100
print("max hash bucket kept:", h.max(), "(should be < 4)")
PY
Pass: every kept loan hashes to bucket 0–3.

2 · Prepare — splits + audit manifest
python -c "import json,gcsfs; print(json.dumps(json.load(gcsfs.GCSFileSystem().open('$RUN/splits.meta.json')),indent=2))"
Pass: seed 42, source SHA-256, loan counts (train 1,749,456 / val 218,682 / test 218,683),
origination ranges ordered train < val < test, split_criterion: loan_stratified_temporal_origination.

Loan-disjoint:

python - <<'PY'
import pandas as pd, os
R=os.environ["RUN"]
tr=set(pd.read_parquet(f"{R}/train.parquet",columns=["loan_id"]).loan_id.unique())
va=set(pd.read_parquet(f"{R}/val.parquet",columns=["loan_id"]).loan_id.unique())
te=set(pd.read_parquet(f"{R}/test.parquet",columns=["loan_id"]).loan_id.unique())
print("train∩val:",len(tr&va)," train∩test:",len(tr&te)," val∩test:",len(va&te)," (all must be 0)")
PY
Pass: all intersections 0.

Temporal cap (pretrain blind to 2023–24) — critical:

python - <<'PY'
import pandas as pd, os
for s in ("train","val","test"):
    d=pd.read_parquet(f"{os.environ['RUN']}/{s}.parquet",columns=["reporting_date"])
    print(s,"max reporting_date:",d.reporting_date.max())
PY
Pass: every split's max reporting_date ≤ 2022-12-31.

3 · Tokenizer v2 — configs/fannie_mae/tokenizer_v2.json
python - <<'PY'
import json,yaml
tok=json.load(open("configs/fannie_mae/tokenizer_v2.json")); sch=tok["config"]
prof=set(sch["profile"]["numeric"]+sch["profile"]["categorical"])
evt=set(sch["event"]["numeric"]+sch["event"]["categorical"])
leak=set(yaml.safe_load(open("configs/fannie_mae/baseline.yaml"))["leakage"])
print("vocab size:",len(tok["vocab"]))
print("leakage cols in FM features?:", (prof|evt)&leak, "(must be empty)")
cal=list(tok["cal"]["categories_"])
print("calendar quarters: min",min(cal),"max",max(cal),"(max must be <= 2022Q4)")
PY
Pass: 552 vocab; empty leakage intersection; cal max = 2022Q4 (confirms fit on capped train).
Round-trip 100% in reports/tokenizer_v2_report.md.

4–5 · Encode — shard manifests
for split in train val; do echo "=== $split ==="; \
python -c "import json,gcsfs; m=json.load(gcsfs.GCSFileSystem().open('$ENC/$split/manifest.json')); \
print('loans',f\"{m['n_loans']:,}\",'| tokens',f\"{m['n_tokens']:,}\",'| shards',m['n_shards'],'| tokenizer',m['config']['tokenizer'])"; done
Pass: train 1,749,456 loans / 1,198,830,016 tokens / 35 shards; val 218,682 / 92,123,892 / 5.
Loan counts equal the prepare splits; tokenizer = tokenizer_v2.json.

Shard structure:

python - <<'PY'
import pandas as pd, os
s=pd.read_parquet(f"{os.environ['ENC']}/train/shard-00000.parquet")
print("cols:",list(s.columns)); r=s.iloc[0]
print("first loan tokens:",r.n_tokens,"events:",r.n_events,
      "| lengths equal:", len(r.input_ids)==len(r.event_index)==len(r.field_type)==len(r.branch))
PY
Pass: columns loan_id,input_ids,event_index,field_type,branch,n_tokens,n_events; four arrays equal length.

6 · Pretrain — runs/m5_full.pt
python - <<'PY'
import torch, fsspec
ck=torch.load(fsspec.open("gs://sriram-credit-fm-data/runs/m5_full.pt","rb").open(),
              map_location="cpu", weights_only=False)
c=ck["config"]; rc=ck.get("run_config",{})
print("dim",c["dim"],"heads",c["n_heads"],"layers",c["profile_layers"],c["event_layers"],c["history_layers"])
print("vocab",c["vocab_size"],"field_types",c["n_field_types"],"steps",ck.get("steps"))
print("best val",ck["history"]["best_val"],"@",ck["history"]["best_step"])
print("train_dir",rc.get("data",{}).get("train_dir"),"| tokenizer",rc.get("tokenizer"))
PY
Pass: dim 384, 8 heads, layers 3/5/6, vocab 552; best val 0.2303 @ 20000; train_dir = run_2000_2022/train, tokenizer = tokenizer_v2.json.

7–9 · OOT fine-tune + baseline
python - <<'PY'
from credit_fm.utils.config import load_config
c=load_config("configs/fannie_mae/finetune_oot.yaml")
print("checkpoint",c.checkpoint,"| tokenizer",c.tokenizer)
print("train cutoffs",c.task.train_cutoffs)
print("test cutoffs",c.task.test_cutoffs,"| horizon",c.task.horizon_months,"| gate",c.task.gate_col)
PY
for f in reports/m5_oot_ft_frozen.md reports/m5_oot_ft_lora.md reports/m5_oot_ft_full.md \
         reports/fannie_oot_2022_2023.md; do echo "--- $f ---"; cat "$f"; done
Pass: checkpoint m5_full.pt, tokenizer v2, train cutoffs 2016–2021, test 2022 & 2023, horizon 12,
gate is_performing. Reports: FM full 0.8257/0.0113, LoRA 0.8068/0.0087, frozen 0.7309/0.0052;
baseline 0.7913/0.0057.

Leakage logic (reason, no command): pretrain capped ≤2022; head trains on cutoffs ≤2021 (labels
≤2022), tested on 2022/2023 cutoffs (labels 2023/2024) — non-overlapping label windows; loans in both
eras hash-split to one side; performing-gate predicts new defaults.

Optional deep check — reproduce a slice
python scripts/finetune.py -c configs/fannie_mae/finetune_oot.yaml \
  --mode frozen --engine vector --runtime.device null --runtime.bf16 true \
  --limit 50000 --report /tmp/m5_ft_frozen_slice.md
Pass: runs clean; prints loan-disjoint / fit / monitor / test / per-epoch val ROC; test ROC in the
same neighbourhood (a 50k slice won't match the full number exactly).

If every check passes: the M5 out-of-time win is validated — model trained provably on ≤2022 data,
tested on unseen 2023–24, leakage controls intact at every stage, beating the same-window baseline on
both metrics.
