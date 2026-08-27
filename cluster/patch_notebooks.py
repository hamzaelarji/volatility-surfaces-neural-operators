#!/usr/bin/env python3
"""
Patch NB03/NB04/NB05 so they can be executed headless on the Telecom cluster
(nbconvert inside tmux) and so that every plotly figure is written to disk
as a static file usable directly in the LaTeX thesis.

Design choice: instead of rewriting the 18 `fig.show()` call sites (risky),
we monkey-patch `plotly.graph_objects.Figure.show` once, in a bootstrap cell
inserted right after the notebook's config cell. Every existing call site is
therefore covered, and the notebook's own logic is untouched.

Idempotent: running it twice does not insert the bootstrap twice.
"""

import sys
import warnings
from pathlib import Path

import nbformat

warnings.filterwarnings("ignore")

MARKER = "THESIS_HEADLESS_BOOTSTRAP_V1"

BOOTSTRAP = '''# === {marker} — inserted for headless cluster runs ===
# Makes the notebook safe to execute with `jupyter nbconvert --execute` on a
# machine with no display, and exports every plotly figure to disk so the
# thesis LaTeX can \\includegraphics them directly.
import os as _os
from pathlib import Path as _Path

import plotly.io as _pio
import plotly.graph_objects as _go

_HEADLESS = _os.environ.get("THESIS_HEADLESS", "0") == "1"

# figure output directory: defaults to <OUT_DIR>/figures, overridable
try:
    _fig_base = OUT_DIR
except NameError:
    _fig_base = _Path("data/clean")
FIG_DIR = _Path(_os.environ.get("THESIS_FIG_DIR", str(_Path(_fig_base) / "figures")))
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIG_PREFIX = _os.environ.get("THESIS_FIG_PREFIX", "{prefix}")
FIG_SHARD = _os.environ.get("NB03_SHARD", "")
FIG_FORMAT = _os.environ.get("THESIS_FIG_FORMAT", "pdf")   # pdf = vector, best for LaTeX
FIG_WRITE_HTML = _os.environ.get("THESIS_FIG_HTML", "1") == "1"

if _HEADLESS:
    # never try to open a browser / require a live kernel front-end
    _pio.renderers.default = "json"
else:
    _pio.renderers.default = _os.environ.get("PLOTLY_RENDERER", "notebook_connected")

_FIG_COUNTER = {{"n": 0}}
_FIG_MANIFEST = []
_orig_fig_show = _go.Figure.show


def _thesis_fig_show(self, *args, **kwargs):
    """Save the figure, then display it only when running interactively."""
    _FIG_COUNTER["n"] += 1
    suffix = f"_{{FIG_SHARD}}" if FIG_SHARD else ""
    stem = f"{{FIG_PREFIX}}{{suffix}}_fig{{_FIG_COUNTER['n']:02d}}"

    saved = []
    try:
        path = FIG_DIR / f"{{stem}}.{{FIG_FORMAT}}"
        self.write_image(str(path), width=1100, height=700, scale=2)
        saved.append(path.name)
    except Exception as exc:                       # kaleido missing / export failure
        print(f"[fig] static export failed for {{stem}} "
              f"({{exc.__class__.__name__}}: {{exc}}) — falling back to HTML")

    if FIG_WRITE_HTML:
        try:
            path = FIG_DIR / f"{{stem}}.html"
            self.write_html(str(path), include_plotlyjs="cdn")
            saved.append(path.name)
        except Exception as exc:
            print(f"[fig] html export failed for {{stem}}: {{exc}}")

    _FIG_MANIFEST.append(dict(stem=stem, files=saved,
                              title=str(getattr(self.layout.title, "text", "") or "")))
    if saved:
        print(f"[fig] {{stem}} -> {{', '.join(saved)}}")

    if not _HEADLESS:
        return _orig_fig_show(self, *args, **kwargs)
    return None


_go.Figure.show = _thesis_fig_show


def write_figure_manifest():
    """Write the figure index so the LaTeX side knows what was produced."""
    try:
        import polars as _plr
        if _FIG_MANIFEST:
            rows = [dict(stem=r["stem"], files=";".join(r["files"]), title=r["title"])
                    for r in _FIG_MANIFEST]
            suffix = f"_{{FIG_SHARD}}" if FIG_SHARD else ""
            out = FIG_DIR / f"{{FIG_PREFIX}}{{suffix}}_figure_manifest.parquet"
            _plr.DataFrame(rows).write_parquet(out)
            print("written:", out)
    except Exception as exc:
        print("[fig] manifest not written:", exc)


print(f"[bootstrap] headless={{_HEADLESS}} | figures -> {{FIG_DIR}} "
      f"(.{{FIG_FORMAT}}) | prefix={{FIG_PREFIX}}{{'_' + FIG_SHARD if FIG_SHARD else ''}}")
'''

EPILOGUE = '''# === {marker} — figure manifest ===
write_figure_manifest()
'''


def find_anchor(nb):
    """Index of the config cell that defines OUT_DIR (bootstrap goes right after)."""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source
        if "OUT_DIR" in src and ("environ" in src or "mkdir" in src):
            return i
    raise RuntimeError("no cell defining OUT_DIR found")


def patch(path: Path, prefix: str) -> None:
    nb = nbformat.read(str(path), as_version=4)

    if any(MARKER in c.source for c in nb.cells):
        print(f"{path.name}: already patched, skipping")
        return

    anchor = find_anchor(nb)
    boot = nbformat.v4.new_code_cell(BOOTSTRAP.format(marker=MARKER, prefix=prefix))
    nb.cells.insert(anchor + 1, boot)
    nb.cells.append(nbformat.v4.new_code_cell(EPILOGUE.format(marker=MARKER)))

    # strip stale outputs and execution counts so the executed notebook is clean
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    nbformat.validate(nb)
    nbformat.write(nb, str(path))
    print(f"{path.name}: bootstrap inserted after cell {anchor}, "
          f"manifest cell appended, outputs cleared")


def main():
    targets = [
        ("03_deep_smoother.ipynb", "nb03"),
        ("04_neural_operator.ipynb", "nb04"),
        ("05_downstream_economics.ipynb", "nb05"),
        ("06_path_dependence_barrier.ipynb", "nb06"),
    ]
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    for name, prefix in targets:
        path = base / name
        if path.exists():
            patch(path, prefix)
        else:
            print(f"{name}: not found, skipped")


if __name__ == "__main__":
    main()
