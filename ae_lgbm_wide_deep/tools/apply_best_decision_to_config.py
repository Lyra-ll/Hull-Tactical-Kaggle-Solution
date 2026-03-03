# tools/apply_best_decision_to_config.py
# ============================================================
# Safely write best decision params (from best_decision_params_v2.json)
# into config.py, ONLY inside an AUTO block.
#
# This version writes PIPELINE-COMPATIBLE keys:
#   DECISION_COEF, DECISION_DECAY, DECISION_THRESHOLD
#   DECISION_CLIP_LOW, DECISION_CLIP_HIGH
#   MIN_INVESTMENT, MAX_INVESTMENT
#   DECISION_USE_WEATHER_REGIME (optional)
#
# It also creates a timestamped .bak backup.
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional


AUTO_START = "# === [AUTO DECISION PARAMS START] ==="
AUTO_END = "# === [AUTO DECISION PARAMS END] ==="

_BLOCK_RE = re.compile(
    re.escape(AUTO_START) + r"[\s\S]*?" + re.escape(AUTO_END),
    flags=re.MULTILINE,
)

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _backup(path: str) -> str:
    bak = f"{path}.{_now_tag()}.bak"
    with open(path, "rb") as fsrc:
        data = fsrc.read()
    with open(bak, "wb") as fdst:
        fdst.write(data)
    return bak

def _py_value(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(int(v))
    if isinstance(v, float):
        return repr(float(v))
    if isinstance(v, str):
        s = v.replace("\\", "\\\\").replace('"', '\\"')
        return f"\"{s}\""
    if v is None:
        return "None"
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{s}\""

def extract_decision_params(best_json: Dict[str, Any], pred_col: Optional[str] = None) -> Dict[str, Any]:
    if "params" in best_json and isinstance(best_json["params"], dict):
        p = best_json["params"]
    else:
        p = best_json

    out: Dict[str, Any] = {}

    # DECISION_DECAY: default 0.0; if best json contains a tuned `decay`, honor it.
    if "decay" in p:
        out["DECISION_DECAY"] = float(p["decay"])
    else:
        out["DECISION_DECAY"] = 0.0

    if "coef" in p: out["DECISION_COEF"] = float(p["coef"])
    if "threshold" in p: out["DECISION_THRESHOLD"] = float(p["threshold"])

    # clip bounds
    if "clip_low" in p: out["DECISION_CLIP_LOW"] = float(p["clip_low"])
    if "clip_high" in p: out["DECISION_CLIP_HIGH"] = float(p["clip_high"])

    # investment bounds used by score_official in pipeline
    if "min_invest" in p: out["MIN_INVESTMENT"] = float(p["min_invest"])
    if "max_invest" in p: out["MAX_INVESTMENT"] = float(p["max_invest"])

    # vol proxy source (optional)
    if "vol_source" in p: out["DECISION_VOL_SOURCE"] = str(p["vol_source"])
    if "vol_window" in p: out["DECISION_VOL_WINDOW"] = int(p["vol_window"])

    # regime switch (optional)
    if "use_regime" in p:
        out["DECISION_USE_WEATHER_REGIME"] = bool(p["use_regime"])

    return out

def build_auto_block(params: Dict[str, Any]) -> str:
    lines = [AUTO_START, f"# auto-generated at: {datetime.now().isoformat(timespec='seconds')}"]
    for k in sorted(params.keys()):
        lines.append(f"{k} = {_py_value(params[k])}")
    lines.append(AUTO_END)
    return "\n".join(lines) + "\n"

def apply_to_config(config_path: str, params: Dict[str, Any], dry_run: bool = False) -> str:
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        txt = f.read()

    new_block = build_auto_block(params).strip("\n")

    if _BLOCK_RE.search(txt):
        new_txt = _BLOCK_RE.sub(new_block, txt)
        if not new_txt.endswith("\n"):
            new_txt += "\n"
    else:
        if not txt.endswith("\n"):
            txt += "\n"
        new_txt = txt + "\n" + new_block + "\n"

    if dry_run:
        return new_txt

    bak = _backup(config_path)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_txt)
    return bak

def apply_best_json_to_config(best_json_path: str, config_path: str, pred_col: Optional[str] = None) -> None:
    with open(best_json_path, "r", encoding="utf-8") as f:
        best = json.load(f)
    params = extract_decision_params(best, pred_col=pred_col)
    apply_to_config(config_path, params, dry_run=False)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_json", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    with open(args.best_json, "r", encoding="utf-8") as f:
        best = json.load(f)
    params = extract_decision_params(best)

    if args.dry_run:
        print(apply_to_config(args.config, params, dry_run=True))
        return 0

    bak = apply_to_config(args.config, params, dry_run=False)
    print(f"[ok] wrote AUTO decision block into {args.config}")
    print(f"[bak] backup -> {bak}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
