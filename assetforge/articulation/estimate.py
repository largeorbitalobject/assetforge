"""Geometric joint-axis / pivot / limits estimator (pure numpy + trimesh, CPU).

This is the metric-geometry half of mechanical rig inference. A VLM (or any other
labeller) tells us the *topology* -- which part is the parent, which is the child,
and the joint *type* (revolute / prismatic). It is NOT trusted for any metric
quantity. Everything below -- the pivot point, the rotation/slide axis, the limits
-- is derived purely from the part meshes.

Why we don't trust a VLM for geometry: an image gives no reliable metric scale,
no occluded interior, and no sub-degree axis direction. Mechanical joints, by
contrast, leave strong geometric fingerprints: the two parts touch along a
contact region whose centroid is the pivot, and the moving part is either
elongated (a barrel -> pitch about the lateral axis) or rotationally symmetric
(a turret head / ring -> yaw about the symmetry axis).

API
---
    estimate_joint(parent_mesh, child_mesh, joint_type) -> dict
        {"origin":[x,y,z], "axis":[ax,ay,az] (unit), "limits":{"min":..,"max":..}}

    estimate_graph(parts, topology) -> list[dict]
        parts: {part_id: trimesh.Trimesh}
        topology: [(parent_id, child_id, joint_type), ...]
        -> [{"id","parent","child","type","origin","axis","limits"}, ...]

Coordinate frame: whatever frame the meshes are passed in. The estimator is
frame-agnostic; the caller is responsible for handing in meshes that live in the
same space as the rig it intends to author (e.g. Z-up Blender space). The world
axes used for snapping are the standard basis of that frame.
"""
from __future__ import annotations

import numpy as np

try:  # scipy is optional; we fall back to a chunked numpy nearest-neighbour.
    from scipy.spatial import cKDTree as _cKDTree
except Exception:  # pragma: no cover
    _cKDTree = None


# --------------------------------------------------------------------------- #
# low-level geometry helpers
# --------------------------------------------------------------------------- #
def _as_verts(mesh) -> np.ndarray:
    """Pull an (N,3) float64 vertex array out of a trimesh.Trimesh (or array-like)."""
    if hasattr(mesh, "vertices"):
        v = np.asarray(mesh.vertices, dtype=np.float64)
    else:
        v = np.asarray(mesh, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"expected (N,3) vertices, got {v.shape}")
    return v


def _nn_dists(a: np.ndarray, b: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """For each point in `a`, distance to the nearest point in `b`.

    Uses scipy.cKDTree when available, otherwise a chunked brute-force numpy
    computation (no scipy dependency required, memory-bounded by `chunk`).
    """
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.inf)
    if _cKDTree is not None:
        d, _ = _cKDTree(b).query(a, k=1)
        return np.asarray(d, dtype=np.float64)
    out = np.empty(len(a), dtype=np.float64)
    b2 = (b * b).sum(1)  # (M,)
    for i in range(0, len(a), chunk):
        ac = a[i : i + chunk]
        # ||ac - b||^2 = |ac|^2 - 2 ac.b + |b|^2
        d2 = (ac * ac).sum(1)[:, None] - 2.0 * ac @ b.T + b2[None, :]
        np.maximum(d2, 0.0, out=d2)
        out[i : i + chunk] = np.sqrt(d2.min(1))
    return out


def _pca(points: np.ndarray):
    """Return (mean, eigvecs, eigvals) with eigvals ASCENDING.

    eigvecs[:, k] is the unit axis with variance eigvals[k]. So eigvecs[:, -1]
    is the long / dominant axis and eigvecs[:, 0] is the thin / normal axis.
    """
    mean = points.mean(0)
    c = points - mean
    cov = (c.T @ c) / max(len(points) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)  # ascending eigenvalues, orthonormal vecs
    return mean, vecs, vals


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _snap_to_world_axis(axis: np.ndarray, tol_deg: float = 15.0) -> np.ndarray:
    """Snap `axis` to the nearest +/- world basis vector if within tol_deg.

    Mechanical parts are usually axis-aligned, and a clean [0,0,1] is far more
    useful downstream than [0.02,-0.01,0.9997]. We only snap when we are
    confident (within tol_deg of a basis axis).
    """
    axis = _unit(np.asarray(axis, dtype=np.float64))
    cos_tol = np.cos(np.deg2rad(tol_deg))
    best_k, best_cos = -1, -1.0
    for k in range(3):
        c = abs(axis[k])  # |dot| with basis vector k
        if c > best_cos:
            best_cos, best_k = c, k
    if best_cos >= cos_tol:
        snapped = np.zeros(3)
        snapped[best_k] = 1.0 if axis[best_k] >= 0 else -1.0
        return snapped
    return axis


def _orient(axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip `axis` so it points into the same half-space as `reference`.

    Axis sign is geometrically arbitrary for a rotation/slide axis, but we make a
    deterministic choice (point from parent toward child) so output is stable.
    """
    axis = _unit(axis)
    if np.dot(axis, reference) < 0:
        axis = -axis
    return axis


# --------------------------------------------------------------------------- #
# contact region
# --------------------------------------------------------------------------- #
def _contact_region(parent_v: np.ndarray, child_v: np.ndarray):
    """Find the vertices of both parts that lie in the mutual contact region.

    Returns (contact_points, child_contact, parent_contact, threshold).

    contact_points stacks the child-side and parent-side contact vertices; its
    centroid is the joint pivot, and its PCA gives the contact-plane geometry.
    """
    d_child = _nn_dists(child_v, parent_v)   # child verts -> nearest parent
    d_parent = _nn_dists(parent_v, child_v)  # parent verts -> nearest child

    # Adaptive distance threshold. Scale by the smaller part's bbox diagonal so
    # the same code works for big and small assemblies. We take vertices within
    # `frac` of that diagonal of the other part.
    def _diag(v):
        return float(np.linalg.norm(v.max(0) - v.min(0)))

    scale = min(_diag(parent_v), _diag(child_v))
    base_thr = 0.05 * scale  # 5% of the smaller part's diagonal

    def _select(dists, verts):
        thr = base_thr
        # If nothing is within base_thr (parts don't visibly touch -- gap, or
        # sampled-vertex coarseness), grow the threshold until we capture a
        # reasonable contact patch (>=5% of verts or 8 verts).
        for _ in range(8):
            mask = dists <= thr
            if mask.sum() >= max(8, int(0.05 * len(verts))):
                break
            thr *= 1.6
        # Final guard: if still empty, take the closest 5% of vertices.
        if mask.sum() == 0:
            k = max(8, int(0.05 * len(verts)))
            idx = np.argsort(dists)[:k]
            mask = np.zeros(len(verts), dtype=bool)
            mask[idx] = True
        return verts[mask], thr

    child_contact, thr_c = _select(d_child, child_v)
    parent_contact, thr_p = _select(d_parent, parent_v)

    contact_points = np.vstack([child_contact, parent_contact])
    return contact_points, child_contact, parent_contact, max(thr_c, thr_p)


# --------------------------------------------------------------------------- #
# axis estimation
# --------------------------------------------------------------------------- #
def _classify_shape(eigvals: np.ndarray):
    """Classify a part by its PCA eigenvalue spectrum (ascending).

    Returns one of "elongated", "disk", "blob".
      elongated: one axis dominates  (lam2 >> lam1, lam2 >> lam0) -> a barrel/rod
      disk     : one axis is tiny    (lam0 << lam1 ~ lam2)        -> a ring/plate
      blob     : roughly isotropic / ambiguous
    """
    lam0, lam1, lam2 = (float(x) for x in eigvals)  # smallest .. largest
    lam2 = max(lam2, 1e-12)
    lam1 = max(lam1, 1e-12)
    elong_ratio = lam2 / lam1          # long vs mid
    flat_ratio = lam0 / lam1           # thin vs mid
    if elong_ratio >= 2.5 and (lam2 / max(lam0, 1e-12)) >= 4.0:
        return "elongated", elong_ratio, flat_ratio
    if flat_ratio <= 0.25:
        return "disk", elong_ratio, flat_ratio
    return "blob", elong_ratio, flat_ratio


def _revolute_axis(parent_v, child_v, contact_points, child_contact, up_hint=None):
    """Estimate the revolute rotation axis using part + contact geometry.

    Strategy, in order of preference:

    1. Classify the CHILD by PCA.
       - "disk"/ring (one tiny eigenvalue): the part is rotationally symmetric
         about its thin axis -> rotation axis = the disk normal = smallest-variance
         PCA axis of the child.  (A turret head/ring yaws about its own normal.)
       - "elongated" (a barrel/rod): the part rotates ABOUT a lateral axis, not
         along its length. The pitch axis is perpendicular to BOTH the long axis
         and the up direction. We derive it as the contact-region spread direction
         that is most orthogonal to the child's long axis -- i.e. the lateral axis
         of the contact patch. This is the key disambiguation for an elevating
         barrel.
       - "blob": fall back to the contact-plane normal.

    2. The contact-plane normal (smallest-variance PCA axis of the contact points)
       is always computed as a strong secondary candidate: a hinge/pivot interface
       is usually a planar patch whose normal is the rotation axis for a yaw-type
       joint, and is perpendicular to the rotation axis for a hinge. We use it to
       break ties and to validate.

    Returns (axis_unit, debug_dict).
    """
    _, cvecs, cvals = _pca(child_v)
    child_long = cvecs[:, -1]   # dominant child axis
    child_norm = cvecs[:, 0]    # thinnest child axis

    # contact-plane geometry
    _, kvecs, kvals = _pca(contact_points)
    contact_normal = kvecs[:, 0]   # thinnest spread of the contact patch
    contact_spread = kvecs[:, -1]  # widest spread of the contact patch

    kind, elong, flat = _classify_shape(cvals)

    dbg = {"child_kind": kind, "elong_ratio": round(elong, 3),
           "flat_ratio": round(flat, 3)}

    if kind == "disk":
        # rotationally symmetric plate/ring -> spin about its own normal
        axis = child_norm
        dbg["rule"] = "disk_normal"
    elif kind == "elongated":
        # A barrel elevates: the pitch axis is perpendicular to the long axis and
        # to up. Build it explicitly when an up hint is available, then refine
        # toward the contact spread which captures the physical hinge width.
        up = _unit(np.asarray(up_hint, dtype=np.float64)) if up_hint is not None else None
        cand = None
        if up is not None and abs(np.dot(_unit(child_long), up)) < 0.97:
            cand = _unit(np.cross(child_long, up))  # lateral = long x up
        # contact lateral: spread direction made orthogonal to the long axis
        spread_perp = contact_spread - np.dot(contact_spread, _unit(child_long)) * _unit(child_long)
        if np.linalg.norm(spread_perp) > 1e-6:
            spread_perp = _unit(spread_perp)
        else:
            spread_perp = None
        # Prefer the cross(long, up) lateral; corroborate with contact spread.
        if cand is not None:
            axis = cand
            dbg["rule"] = "elongated_long_x_up"
            if spread_perp is not None:
                dbg["spread_agree"] = round(abs(float(np.dot(cand, spread_perp))), 3)
        elif spread_perp is not None:
            axis = spread_perp
            dbg["rule"] = "elongated_contact_lateral"
        else:
            axis = contact_normal
            dbg["rule"] = "elongated_fallback_contact_normal"
    else:  # blob -- least informative; trust the contact patch
        axis = contact_normal
        dbg["rule"] = "blob_contact_normal"

    return _unit(axis), dbg


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def estimate_joint(parent_mesh, child_mesh, joint_type, up_hint=None) -> dict:
    """Estimate origin / axis / limits for a single joint from the two meshes.

    Parameters
    ----------
    parent_mesh, child_mesh : trimesh.Trimesh (or (N,3) vertex arrays)
    joint_type              : "revolute" | "prismatic" | "fixed"
    up_hint                 : optional [x,y,z] world up; helps disambiguate the
                              lateral (pitch) axis of an elongated child. Defaults
                              to +Z (Blender / graph convention).

    Returns
    -------
    {"origin":[x,y,z], "axis":[ax,ay,az], "limits":{"min","max"},
     "limits_are_default": True, "debug": {...}}
    """
    parent_v = _as_verts(parent_mesh)
    child_v = _as_verts(child_mesh)
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0])
    up_hint = _unit(np.asarray(up_hint, dtype=np.float64))

    # --- pivot: centroid of the mutual contact region -----------------------
    contact_points, child_contact, parent_contact, thr = _contact_region(parent_v, child_v)
    origin = contact_points.mean(0)

    # reference direction: parent centroid -> child centroid (for sign + slide)
    p2c = _unit(child_v.mean(0) - parent_v.mean(0))

    debug = {"n_contact": int(len(contact_points)), "contact_thr": round(float(thr), 5)}

    if joint_type == "revolute":
        axis_raw, dbg = _revolute_axis(parent_v, child_v, contact_points,
                                       child_contact, up_hint=up_hint)
        debug.update(dbg)
        # orient: for a yaw-type axis, prefer alignment with up; otherwise stay
        # consistent by pointing into the parent->child half-space.
        if abs(np.dot(axis_raw, up_hint)) > 0.7:
            axis = _orient(axis_raw, up_hint)
        else:
            axis = _orient(axis_raw, p2c if np.dot(axis_raw, p2c) != 0 else up_hint)
        axis = _snap_to_world_axis(axis)
        limits = {"min": -float(np.pi), "max": float(np.pi)}

    elif joint_type == "prismatic":
        # slide direction = the child's dominant PCA long axis
        _, cvecs, _ = _pca(child_v)
        axis = _orient(cvecs[:, -1], p2c)
        axis = _snap_to_world_axis(axis)
        # default travel = bbox extent of the child along the (snapped) axis
        proj = child_v @ axis
        extent = float(proj.max() - proj.min())
        limits = {"min": 0.0, "max": extent}
        debug["rule"] = "prismatic_long_axis"
        debug["bbox_extent_along_axis"] = round(extent, 5)

    elif joint_type == "fixed":
        axis = np.array([0.0, 0.0, 1.0])
        limits = {"min": 0.0, "max": 0.0}
        debug["rule"] = "fixed"

    else:
        raise ValueError(f"unknown joint_type {joint_type!r}")

    return {
        "origin": [float(x) for x in origin],
        "axis": [float(x) for x in axis],
        "limits": limits,
        "limits_are_default": True,  # limits cannot be inferred from one static mesh
        "debug": debug,
    }


def estimate_graph(parts: dict, topology: list, up_hint=None) -> list:
    """Estimate all joints for an articulation graph.

    Parameters
    ----------
    parts    : {part_id: trimesh.Trimesh}
    topology : [(parent_id, child_id, joint_type), ...]
    up_hint  : optional world up vector (default +Z).

    Returns a list of joint dicts:
       {"id","parent","child","type","origin","axis","limits", ...}
    The joint id is "<parent>_<child>".
    """
    joints = []
    for parent_id, child_id, jtype in topology:
        if parent_id not in parts:
            raise KeyError(f"parent {parent_id!r} not in parts")
        if child_id not in parts:
            raise KeyError(f"child {child_id!r} not in parts")
        est = estimate_joint(parts[parent_id], parts[child_id], jtype, up_hint=up_hint)
        joints.append({
            "id": f"{parent_id}_{child_id}",
            "parent": parent_id,
            "child": child_id,
            "type": jtype,
            "origin": est["origin"],
            "axis": est["axis"],
            "limits": est["limits"],
            "limits_are_default": est["limits_are_default"],
            "debug": est["debug"],
        })
    return joints


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _load_turret_parts(glb_path):
    """Load the turret fixture, splitting into named parts in Z-up graph space.

    The GLB is authored Y-up (glTF convention); graph.json is Z-up. We rotate
    +90 deg about X (which maps (x,y,z) -> (x,-z,y)) so the loaded meshes match
    the ground-truth graph frame.
    """
    import trimesh
    scene = trimesh.load(glb_path, process=False)
    y_to_z = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    name_map = {"base": "base", "head": "head", "barrel": "barrel"}
    parts = {}
    for node in scene.graph.nodes_geometry:
        T, geom = scene.graph[node]
        g = scene.geometry[geom].copy()
        g.apply_transform(T)         # node-local -> scene world (Y-up)
        g.apply_transform(y_to_z)    # Y-up -> Z-up (graph frame)
        if node in name_map:
            parts[name_map[node]] = g
    return parts


def _self_test():
    import json
    import trimesh  # noqa: F401  (import here so the module imports w/o trimesh)

    glb = "/home/orbital/.claude/jobs/fde14dc0/tmp/m0_turret/turret_parts.glb"
    gt_path = "/home/orbital/.claude/jobs/fde14dc0/tmp/m0_turret/graph.json"
    gt = json.load(open(gt_path))
    gt_joints = {j["id"]: j for j in gt["joints"]}

    parts = _load_turret_parts(glb)
    print("loaded parts:", {k: tuple(v.vertices.shape) for k, v in parts.items()})

    topology = [("base", "head", "revolute"), ("head", "barrel", "revolute")]
    est = estimate_graph(parts, topology, up_hint=[0, 0, 1])

    # map estimated joints to GT joints by parent/child
    gt_by_pc = {(j["parent"], j["child"]): j for j in gt["joints"]}

    all_pass = True
    print("\n=== SELF-TEST: turret ===")
    for ej in est:
        pc = (ej["parent"], ej["child"])
        gj = gt_by_pc[pc]
        true_axis = _unit(np.asarray(gj["axis"], dtype=np.float64))
        true_origin = np.asarray(gj["origin"], dtype=np.float64)
        est_axis = np.asarray(ej["axis"], dtype=np.float64)
        est_origin = np.asarray(ej["origin"], dtype=np.float64)

        adot = abs(float(np.dot(est_axis, true_axis)))
        ang = np.rad2deg(np.arccos(min(adot, 1.0)))
        pdist = float(np.linalg.norm(est_origin - true_origin))

        axis_ok = adot > 0.97
        pivot_ok = pdist < 0.25
        all_pass = all_pass and axis_ok

        status = "PASS" if axis_ok else "FAIL"
        print(f"\n[{status}] joint {gj['id']}  ({pc[0]} -> {pc[1]}, {ej['type']})")
        print(f"   est axis   {np.round(est_axis,4).tolist()}   true {gj['axis']}")
        print(f"   |dot|      {adot:.4f}  ({ang:.2f} deg)   axis_ok(>0.97)={axis_ok}")
        print(f"   est origin {np.round(est_origin,4).tolist()}   true {gj['origin']}")
        print(f"   pivot dist {pdist:.4f}                  pivot_ok(<0.25)={pivot_ok}")
        print(f"   rule       {ej['debug'].get('rule')}   debug={ej['debug']}")

    print("\n=== RESULT:", "ALL AXES PASS" if all_pass else "SOME AXES FAIL", "===")
    return all_pass


if __name__ == "__main__":
    import sys
    ok = _self_test()
    sys.exit(0 if ok else 1)
