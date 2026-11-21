"""Auto-detect skeleton ROLES from a RigAnything-style skeleton (joints + parents).

Replaces the hardcoded PELVIS=0 / SPINE=[1,2,3] / leg-index literals so animation generalizes
to arbitrary creatures. Pure geometry/topology heuristics (no ML). Output feeds graph.json["roles"]
and the generalized walk. Honest ceiling: tuned for biped/quadruped-ish morphologies; exotic
topologies (hexapods, wings-as-arms) may mislabel — see docs/PHASE3-RIGGING.md.

Roles schema:
  { "pelvis": int, "spine": [int...], "neck": int|None, "head": int|None,
    "legs": [{"side":"L|R|C","hip","knee","ankle","toe"} ...],
    "arms": [{"side":"L|R|C","shoulder","elbow","wrist"} ...],
    "tail": [int...] }
Indices are flip-invariant, so they index the raw RigAnything joint order directly.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def _to_zup(joints: np.ndarray) -> np.ndarray:
    # RigAnything-swapped -> Blender Z-up = (x, -y, -z)  (matches build_rig.py)
    J = np.asarray(joints, float).copy()
    J[:, 1] *= -1; J[:, 2] *= -1
    return J


def classify_roles(joints: np.ndarray, parents) -> dict:
    J = _to_zup(joints)
    z, x = J[:, 2], J[:, 0]
    zmin, zmax = z.min(), z.max(); H = max(zmax - zmin, 1e-6)
    relz = (z - zmin) / H
    XR = max(float(np.max(np.abs(x))), 1e-6)   # skeleton's own lateral extent (adaptive thresholds)
    n = len(J)
    parents = [int(p) for p in parents]

    ch = defaultdict(list)
    root = 0
    for i in range(n):
        p = parents[i]
        if p == i or p < 0:
            root = i
        else:
            ch[p].append(i)

    def descendants(i):
        out, stack = [], list(ch[i])
        while stack:
            k = stack.pop(); out.append(k); stack += ch[k]
        return out

    def subtree_minz(i):
        return min([relz[i]] + [relz[d] for d in descendants(i)])

    def subtree_maxz(i):
        return max([relz[i]] + [relz[d] for d in descendants(i)])

    def subtree_maxabsx(i):
        return max([abs(x[i])] + [abs(x[d]) for d in descendants(i)])

    def path_to_extreme(start, low=True):
        """Greedily follow the child whose subtree reaches the most extreme z."""
        path = [start]
        while ch[path[-1]]:
            kids = ch[path[-1]]
            key = (lambda k: subtree_minz(k)) if low else (lambda k: subtree_maxz(k))
            path.append(min(kids, key=key) if low else max(kids, key=key))
        return path

    def pick4(path):
        """Reduce a chain to (root, ~1/3, ~2/3, tip) = hip/knee/ankle/toe-ish."""
        if len(path) <= 4:
            idx = path + [path[-1]] * (4 - len(path))
        else:
            L = len(path) - 1
            idx = [path[0], path[round(L / 3)], path[round(2 * L / 3)], path[-1]]
        return idx

    pelvis = root
    roles = {"pelvis": pelvis, "spine": [], "neck": None, "head": None,
             "legs": [], "arms": [], "tail": []}

    # --- pelvis child subtrees: classify into legs / spine-trunk / tail ---
    spine_seed = None; spine_top = -1
    for c in ch[pelvis]:
        smin, smax = subtree_minz(c), subtree_maxz(c)
        reaches_ground = smin < 0.18
        goes_up = smax > relz[pelvis] + 0.08
        if reaches_ground and not goes_up:
            # leg OR tail. leg = lateral terminal near ground; tail = central, mid-low, no ground-lateral foot
            path = path_to_extreme(c, low=True)
            tip = path[-1]
            if abs(x[tip]) > 0.15 * XR or relz[tip] < 0.1:   # lateral foot or at the floor => leg
                hip, knee, ankle, toe = pick4(path)
                side = "L" if np.mean([x[i] for i in path]) < 0 else "R"
                roles["legs"].append({"side": side, "hip": hip, "knee": knee,
                                      "ankle": ankle, "toe": toe})
            else:
                roles["tail"] = path
        else:
            # candidate spine trunk (goes up / central)
            if smax > spine_top:
                spine_top = smax; spine_seed = c

    # --- spine trunk: central upward chain to the shoulder hub ---
    if spine_seed is not None:
        spine = [pelvis, spine_seed]
        cur = spine_seed
        hub = None
        while ch[cur]:
            kids = ch[cur]
            # hub = node with >=2 children whose SUBTREES splay laterally (arm chains start
            # central and only splay further out, so test subtree extent, not the child's x)
            lateral_kids = [k for k in kids
                            if subtree_maxabsx(k) > 0.45 * XR and subtree_minz(k) > 0.2]
            if len(lateral_kids) >= 2:
                hub = cur
                break
            nxt = max(kids, key=subtree_maxz)   # keep climbing the tallest
            spine.append(nxt); cur = nxt
        roles["spine"] = spine

        hub_node = hub if hub is not None else cur
        arm_seeds = [k for k in ch[hub_node]
                     if subtree_maxabsx(k) > 0.45 * XR and subtree_minz(k) > 0.2]
        central = [k for k in ch[hub_node] if k not in arm_seeds]

        # --- neck/head: the central child climbing from the hub ---
        if central:
            neck = max(central, key=subtree_maxz)
            neck_path = path_to_extreme(neck, low=False)
            roles["neck"] = neck
            roles["head"] = neck_path[-1]
            roles["spine"] = [s for s in spine if s not in neck_path]

        # --- arms: the hub's laterally-splaying children ---
        for k in arm_seeds:
            apath = path_to_extreme(k, low=False)
            wrist = max(apath, key=lambda i: len(ch[i]))   # finger hub = hand
            if len(ch[wrist]) < 2:
                wrist = apath[-1]
            wi = apath.index(wrist)
            elbow = apath[max(1, wi // 2)] if wi >= 1 else apath[min(1, len(apath) - 1)]
            side = "L" if x[k] < 0 else "R"
            roles["arms"].append({"side": side, "shoulder": k, "elbow": elbow, "wrist": wrist})


    # --- assign sides by relative x ordering (robust to off-center skeletons) ---
    def assign_sides(items, key):
        if len(items) < 2:
            for it in items: it["side"] = "C"
            return
        order = sorted(range(len(items)), key=lambda i: x[items[i][key]])
        half = len(order) / 2.0
        for rank, i in enumerate(order):
            items[i]["side"] = "L" if rank < half else "R"
    assign_sides(roles["legs"], "hip")
    assign_sides(roles["arms"], "shoulder")
    return roles


def height_zup(joints: np.ndarray) -> float:
    z = _to_zup(joints)[:, 2]
    return float(z.max() - z.min())


if __name__ == "__main__":
    import sys, json
    npz = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/orbital/projects/orbitis-station/tools/RigAnything/outputs/minotaur/inputs_minotaur.npz"
    d = np.load(npz, allow_pickle=True)
    r = classify_roles(d["joints"], d["parents"])
    print(json.dumps(r, indent=2, default=int))
    print("H =", round(height_zup(d["joints"]), 3))
