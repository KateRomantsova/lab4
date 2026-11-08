"""Частина D. Панорама з 5 кадрів (кроки 13-16): граф зіставлень -> дерево -> опорний кадр -> гомографії."""
import csv, os
import numpy as np, cv2
from common import *

T_RATIO = 0.75
TAU = 3.0
CAP = 5000          # максимальний розмір полотна (px); при більшому — масштабуємо


def feather_weight(mask):
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return d / (d.max() + 1e-9)


def pairwise_inliers(frames):
    """Матриця кількості узгоджених пар між усіма кадрами (швидкий cv2.RANSAC) — для вибору зв'язків."""
    n = len(frames)
    feats = [sift_features(f) for f in frames]
    M = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i + 1, n):
            k1, d1 = feats[i]; k2, d2 = feats[j]
            m = mutual_filter(ratio_filter(knn_pairs(d1, d2), T_RATIO), d1, d2)
            if len(m) >= 8:
                s, d = pts_from_matches(m, k1, k2)
                _, mk = cv2.findHomography(s, d, cv2.RANSAC, TAU)
                M[i, j] = M[j, i] = int(mk.sum()) if mk is not None else 0
    return M


def max_spanning_tree(M):
    n = len(M); inT = {0}; edges = []
    while len(inT) < n:
        best = max(((M[i, j], i, j) for i in inT for j in range(n) if j not in inT))
        edges.append((best[1], best[2])); inT.add(best[2])
    return edges


def tree_center(n, edges):
    adj = {i: [] for i in range(n)}
    for a, b in edges: adj[a].append(b); adj[b].append(a)
    def ecc(r):
        dist = {r: 0}; q = [r]
        for u in q:
            for v in adj[u]:
                if v not in dist: dist[v] = dist[u] + 1; q.append(v)
        return max(dist.values())
    return min(range(n), key=lambda r: (ecc(r), -len(adj[r]))), adj


def run(frames=None, tag="d", seed0=0, save=True):
    if frames is None:
        frames, _ = load_frames()
    n = len(frames)
    M = pairwise_inliers(frames)
    print("матриця узгоджених пар:\n", M)
    if save: np.savetxt(os.path.join(RES, f"{tag}_matrix.csv"), M, fmt="%d", delimiter=",")
    edges = max_spanning_tree(M)
    ref, adj = tree_center(n, edges)
    print("ребра дерева:", [(a + 1, b + 1, int(M[a, b])) for a, b in edges], "опорний кадр:", ref + 1)

    # крок 13: від опорного кадру вглиб дерева: T_child = T_parent · H_{child -> parent}
    T = {ref: np.eye(3)}; stats = []; q = [ref]
    for u in q:
        for v in adj[u]:
            if v in T: continue
            H, src, dst, mask = pair_homography(frames[v], frames[u], T_RATIO, TAU, 2000, seed=seed0 + v)   # src = кадр v, dst = кадр u
            e = reproj_errors(H, src, dst)[mask]
            T[v] = T[u] @ H
            stats.append((v + 1, u + 1, len(src), int(mask.sum()), float(e.mean())))
            print(f"кадр {v+1} -> кадр {u+1}: пар {len(src)}, узгоджених {mask.sum()}, похибка {e.mean():.3f} px")
            q.append(v)
    if save:
        with open(os.path.join(RES, f"{tag}_pairs.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["frame", "to_frame", "matches", "inliers", "mean_err_px"])
            for s in stats: w.writerow([s[0], s[1], s[2], s[3], round(s[4], 3)])
    allinl = sum(s[3] for s in stats)
    wmean = sum(s[3] * s[4] for s in stats) / allinl            # крок 16
    print(f"середня похибка по всіх парах (зважена): {wmean:.3f} px")

    # крок 14: розмір полотна
    allc = []
    for i, im in enumerate(frames):
        h, w = im.shape[:2]
        allc.append(transfer(T[i], np.float32([[0, 0], [w, 0], [w, h], [0, h]])))
    allc = np.vstack(allc)
    xmin, ymin = np.floor(allc.min(0)).astype(int); xmax, ymax = np.ceil(allc.max(0)).astype(int)
    shift = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], np.float64)
    Wc, Hc = int(xmax - xmin), int(ymax - ymin)
    sc = min(1.0, CAP / max(Wc, Hc))
    S = np.diag([sc, sc, 1.0])
    Wc2, Hc2 = int(Wc * sc), int(Hc * sc)
    print(f"полотно {Wc}x{Hc} (масштаб {sc:.3f} -> {Wc2}x{Hc2}), зсув ({-xmin}, {-ymin})")

    area_ratio = Wc * Hc / (n * frames[0].shape[0] * frames[0].shape[1])
    if not save:
        return dict(n=n, canvas=(Wc, Hc), area_ratio=area_ratio, mean_err=wmean, inliers=allinl)
    # крок 15: перенесення і лінійний перехід
    acc = np.zeros((Hc2, Wc2, 3), np.float64); wsum = np.zeros((Hc2, Wc2), np.float64)
    for i, im in enumerate(frames):
        Mi = S @ shift @ T[i]
        warped = cv2.warpPerspective(im, Mi, (Wc2, Hc2)).astype(np.float64)
        msk = cv2.warpPerspective(np.ones(im.shape[:2], np.uint8), Mi, (Wc2, Hc2)) > 0
        wgt = feather_weight(msk) * msk
        acc += warped * wgt[..., None]; wsum += wgt
    pano = (acc / np.maximum(wsum, 1e-9)[..., None]).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(FIG, f"{tag}_panorama.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(os.path.join(RES, f"{tag}_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ref_frame", "canvas_w", "canvas_h", "scale", "weighted_mean_err_px", "total_inliers"])
        w.writerow([ref + 1, Wc, Hc, round(sc, 3), round(wmean, 3), allinl])
    return dict(n=n, ref=ref + 1, canvas=(Wc, Hc), area_ratio=area_ratio, mean_err=wmean, inliers=allinl, matrix=M, edges=[(a + 1, b + 1, int(M[a, b])) for a, b in edges], stats=stats)


if __name__ == "__main__":
    r = run()
    print("відношення площі полотна до сумарної площі кадрів: %.1f" % r["area_ratio"])
