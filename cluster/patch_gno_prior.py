#!/usr/bin/env python3
"""Add NB04_SKIP_GNO_PRIOR so gno_prior can be skipped while keeping deeponet_prior."""
import sys, warnings, nbformat
warnings.filterwarnings("ignore")

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)

OLD = '''    pgno_hist, pgno_res, pgno_models = run_family(
        PriorGNO, "gno_prior", lr_pre=1e-3, lr_real=1e-3, lr_ft=5e-4,
        pretrain_bank=synth_bank_pert)
    all_histories += pgno_hist; all_results += pgno_res
    trained_models["gno_prior"] = pgno_models'''

NEW = '''    if os.environ.get("NB04_SKIP_GNO_PRIOR", "0") == "1":
        print("[prior] NB04_SKIP_GNO_PRIOR=1 -- gno_prior training/export skipped")
    else:
        pgno_hist, pgno_res, pgno_models = run_family(
            PriorGNO, "gno_prior", lr_pre=1e-3, lr_real=1e-3, lr_ft=5e-4,
            pretrain_bank=synth_bank_pert)
        all_histories += pgno_hist; all_results += pgno_res
        trained_models["gno_prior"] = pgno_models'''

if any("NB04_SKIP_GNO_PRIOR" in c.source for c in nb.cells):
    print("already patched"); sys.exit(0)

hits = 0
for c in nb.cells:
    if c.cell_type == "code" and OLD in c.source:
        c.source = c.source.replace(OLD, NEW); hits += 1

if hits != 1:
    sys.exit(f"FAILED: expected 1 match, found {hits} — notebook not modified")

nbformat.validate(nb)
nbformat.write(nb, path)
print("patched OK: NB04_SKIP_GNO_PRIOR is now available")
