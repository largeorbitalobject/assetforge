"""M2: infer a mechanical articulation graph from a parts-segmented mesh.

parts (named submeshes) -> contact adjacency -> kinematic tree (largest/lowest = fixed base) ->
per-joint type/axis/pivot via the geometric estimator -> ArticulationGraph (-> build_artic.py).

VLM (MiniCPM-o) can refine joint TYPE/topology when available; default heuristic: base is fixed,
all other joints revolute (axis/pivot from geometry). Joint motion presets assigned by simple rules.
"""
from __future__ import annotations
import numpy as np
from .graph import ArticulationGraph, Part, Joint
from . import estimate as EST


def _contact_count(a, b, thr):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(b.vertices).query(a.vertices, k=1)
    return int((d < thr).sum())


def infer_graph(parts: dict, mesh_path: str, vlm=None, up_idx: int = 1) -> ArticulationGraph:
    """parts: {name: trimesh.Trimesh}. up_idx = up axis (1=Y for GLB, 2=Z). ArticulationGraph (mechanical)."""
    names = list(parts)
    # scene scale for contact threshold
    allv = np.vstack([np.asarray(parts[n].vertices) for n in names])
    scale = float(np.ptp(allv, 0).max())
    thr = 0.03 * scale

    # base = largest part by vertex count, tie-break lowest centroid z
    # contact adjacency (used by both VLM and heuristic paths)
    adj = {n: [] for n in names}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if _contact_count(parts[a], parts[b], thr) > 0 or _contact_count(parts[b], parts[a], thr) > 0:
                adj[a].append(b); adj[b].append(a)

    def low(n):
        return float(np.asarray(parts[n].vertices)[:, up_idx].mean())

    parent, jtypes, jmotion, base = {}, {}, {}, None
    # --- VLM topology (Qwen3-VL): type + parent/child; geometry still owns axis/pivot ---
    if vlm is not None:
        try:
            pdesc = []
            for n in names:
                v = np.asarray(parts[n].vertices)
                pdesc.append({"name": n,
                              "centroid": [round(float(x), 3) for x in v.mean(0)],
                              "size": [round(float(x), 3) for x in np.ptp(v, 0)],
                              "contacts": adj[n]})
            # VISION: render parts color-coded and let the VLM SEE the structure
            imgs, legend = None, None
            try:
                import subprocess, os, json as _json
                from ..rocm_env import BLENDER, gpu_env
                from pathlib import Path
                cdir = os.path.join(os.path.dirname(mesh_path) or ".", "_partcolor")
                env = gpu_env(); env.pop("LD_LIBRARY_PATH", None)
                subprocess.run([BLENDER, "-b", "-noaudio", "-P",
                                str(Path(__file__).parent.parent / "blender_scripts" / "render_parts_colored.py"),
                                "--", str(mesh_path), cdir], env=env, check=True, capture_output=True, text=True, timeout=300)
                legend = _json.load(open(os.path.join(cdir, "legend.json")))
                imgs = [os.path.join(cdir, v) for v in ("front.png", "side.png", "threeq.png") if os.path.exists(os.path.join(cdir, v))]
            except Exception as e:
                print("part-color render failed:", str(e)[:120])
            topo = vlm.joint_topology(pdesc, images=imgs, legend=legend)
            if topo.get("base") in names:
                base = topo["base"]
                for j in topo.get("joints", []):
                    if j.get("child") in names and (j.get("parent") in names):
                        parent[j["child"]] = j["parent"]
                        jtypes[j["child"]] = j.get("type", "revolute")
                        jmotion[j["child"]] = j.get("motion", "scan")
        except Exception as e:
            print("VLM topology failed, heuristic fallback:", str(e)[:120])

    # --- heuristic fallback / completion ---
    if base is None:
        lo = min(low(n) for n in names); span = max(scale, 1e-6)
        base = max(names, key=lambda n: (low(n) < lo + 0.15 * span, len(parts[n].vertices)))
    parent[base] = None
    if len([n for n in parent if parent[n] is not None]) < len(names) - 1:
        seen = {base}; q = [base]
        while q:
            cur = q.pop(0)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb); parent.setdefault(nb, cur); q.append(nb)
    for n in names:
        if n not in parent:
            parent[n] = base

    g = ArticulationGraph(kind="mechanical", mesh=mesh_path)
    for n in names:
        m = parts[n]; v = np.asarray(m.vertices)
        g.parts.append(Part(id=n, object=n, centroid=[float(x) for x in v.mean(0)],
                            bbox=[[float(x) for x in v.min(0)], [float(x) for x in v.max(0)]]))
    for n in names:
        p = parent[n]
        if p is None:
            continue
        jtype = jtypes.get(n, "revolute")
        est = EST.estimate_joint(parts[p], parts[n], jtype)
        axis = est["axis"]
        preset = jmotion.get(n) or ("scan" if abs(axis[up_idx]) > 0.7 else "elevate")
        g.joints.append(Joint(id=f"{p}_{n}", parent=p, child=n, type=jtype,
                              origin=est["origin"], axis=axis, limits=est.get("limits"),
                              motion={"preset": preset}))
    return g


def load_parts_from_glb(glb_path) -> dict:
    """If a GLB already has named part objects (e.g. the turret fixture), load them as a dict."""
    import trimesh
    s = trimesh.load(glb_path, process=False)
    if isinstance(s, trimesh.Scene):
        return {name: geom for name, geom in s.geometry.items()}
    return {"part0": s}


if __name__ == "__main__":
    import sys, json
    glb = sys.argv[1]
    parts = load_parts_from_glb(glb)
    print("parts:", list(parts))
    g = infer_graph(parts, glb)
    errs = g.validate()
    print("graph kind", g.kind, "parts", len(g.parts), "joints", len(g.joints), "errs", errs)
    for j in g.joints:
        print(f"  joint {j.id}: {j.type} axis={[round(x,2) for x in j.axis]} origin={[round(x,2) for x in j.origin]} motion={j.motion}")
