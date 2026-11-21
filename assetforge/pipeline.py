"""Stage orchestrator: runs the type's stages in order, resumable via the manifest."""
from __future__ import annotations
from typing import Callable

from . import config as cfgmod
from . import workspace as wsmod
from .stages import REGISTRY


def run(image: str, asset_type: str, *, overrides: dict | None = None,
        resume: bool = True, until: str | None = None,
        only: str | None = None, log: Callable[[str], None] = print) -> dict:
    """Execute the pipeline for one image. Returns the final manifest's stage outputs."""
    cfg = cfgmod.load(asset_type, overrides)
    ws = wsmod.Workspace.create(image, asset_type)
    log(f"[assetforge] {asset_type}  ->  {ws.root}")

    stages = list(cfg.stages)
    if only:
        stages = [only]
    elif until:
        if until not in stages:
            raise ValueError(f"--until {until!r} not in {stages}")
        stages = stages[: stages.index(until) + 1]

    upstream: dict[str, dict] = {
        s: ws.outputs(s) for s in cfg.stages if ws.outputs(s)
    }

    order = list(cfg.stages)
    for stage in stages:
        # fingerprint depends only on cfg + outputs of stages STRICTLY BEFORE this one
        prior = order[: order.index(stage)]
        prior_outputs = {s: upstream.get(s) for s in prior}
        fp = wsmod.fingerprint(cfgmod.as_dict(cfg), prior_outputs)
        if resume and not only and ws.is_fresh(stage, fp):
            log(f"  = {stage}: up-to-date (skip)")
            upstream[stage] = ws.outputs(stage)
            continue
        log(f"  > {stage}: running")
        try:
            outputs = REGISTRY[stage](ws, cfg, upstream, log)
        except NotImplementedError as e:
            log(f"  ! {stage}: not implemented — {e}")
            ws.record(stage, fp, {}, status="skipped")
            break
        ws.record(stage, fp, outputs)
        upstream[stage] = outputs
        log(f"  ✓ {stage}: {', '.join(outputs) or '(no outputs)'}")

    return ws.manifest["stages"]
