"""Частина E. Випадки відмови (крок 17) на власних знімках:
 E1 — мале перекриття (зменшуємо перекриття відомої пари обрізанням кадру, еталон — H з усієї пари);
 E2 — періодичний візерунок (бруківка);
 E3 — ланцюжок із 4 і 5 кадрів: геометрично неможлива гомографічна панорама."""
import csv, os
import numpy as np, cv2
from common import *
import part_d


def overlap_fraction(H, shape1, shape2):
    h1, w1 = shape1[:2]; h2, w2 = shape2[:2]
    m = cv2.warpPerspective(np.ones((h1, w1), np.uint8), H, (w2, h2))      # частина кадру 1 всередині кадру 2
    inv = cv2.warpPerspective(np.ones((h2, w2), np.uint8), np.linalg.inv(H), (w1, h1))   # те саме у координатах кадру 1
    return inv.sum() / (w1 * h1)


def estimate(i1, i2, T=0.75, tau=3.0):
    k1, d1 = sift_features(i1); k2, d2 = sift_features(i2)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return None, 0, 0, None, None
    m = mutual_filter(ratio_filter(knn_pairs(d1, d2), T), d1, d2)
    if len(m) < 4:
        return None, len(m), 0, None, None
    src, dst = pts_from_matches(m, k1, k2)
    H, mask = ransac_homography(src, dst, tau, 2000, seed=0)
    return H, len(m), int(mask.sum()), (k1, k2, m, mask), (src, dst)


def run():
    img1, img2 = load_pair()
    H_gt, src, dst, mask = pair_homography(img1, img2, 0.75, 3.0, 2000, 0)
    print(f"еталонна H: узгоджених {mask.sum()}, похибка {reproj_errors(H_gt, src, dst)[mask].mean():.3f} px, перекриття {100*overlap_fraction(H_gt, img1.shape, img2.shape):.0f} %")
    h, w = img1.shape[:2]

    # ---------- E1: мале перекриття. Відрізаємо від кадру 1 смугу c px з боку, де перекриття (ліворуч)
    def cropped(c):
        S = np.array([[1, 0, -c], [0, 1, 0], [0, 0, 1]], float)            # координати кадру 1 -> координати обрізка
        Hc = H_gt @ np.linalg.inv(S)                                        # обрізок -> кадр 2
        return img1[:, c:], Hc
    best = {}
    for c in range(0, w - 80, 4):
        im, Hc = cropped(c)
        ov = 100 * overlap_fraction(Hc, im.shape, img2.shape)
        for tgt in (40, 15, 10, 5):
            if tgt not in best or abs(ov - tgt) < abs(best[tgt][1] - tgt):
                best[tgt] = (c, ov)
    rows = []
    for tgt in (40, 15, 10, 5):
        c, ov = best[tgt]
        im, Hc = cropped(c)
        He, ncand, ninl, _, _ = estimate(im, img2)
        err = float("nan")
        if He is not None:
            gx, gy = np.meshgrid(np.linspace(0, im.shape[1] - 1, 25), np.linspace(0, h - 1, 15))
            g = np.float32(np.c_[gx.ravel(), gy.ravel()])
            tg = transfer(Hc, g); ok = (tg[:, 0] >= 0) & (tg[:, 0] < w) & (tg[:, 1] >= 0) & (tg[:, 1] < h)       # лише точки зони перекриття
            if ok.sum() > 0:
                err = float(np.linalg.norm(transfer(He, g[ok]) - tg[ok], axis=1).mean())
        rows.append((tgt, round(ov, 1), im.shape[1], ncand, ninl, err))
        print(f"E1 перекриття ~{tgt}% (виміряне {ov:.1f}%): кандидатів {ncand}, узгоджених {ninl}, похибка {err:.2f} px")
    with open(os.path.join(RES, "e1_overlap.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["target_overlap", "measured_overlap", "crop_width", "candidates", "inliers", "err_vs_ref_px"])
        for r in rows: wr.writerow([r[0], r[1], r[2], r[3], r[4], "" if np.isnan(r[5]) else round(r[5], 2)])

    # ---------- E2: періодичний візерунок (бруківка). Беремо нижні смуги кадрів пари (лише плитка), еталон — H з усієї пари
    rows2 = []
    for y0f in (0.0, 0.60, 0.72, 0.80):
        y0 = int(y0f * h)
        S = np.array([[1, 0, 0], [0, 1, -y0], [0, 0, 1]], float)
        Hc = S @ H_gt @ np.linalg.inv(S)
        a_, b_ = img1[y0:], img2[y0:]
        He, ncand, ninl, vis, _ = estimate(a_, b_)
        err = float("nan")
        if He is not None:
            gx, gy = np.meshgrid(np.linspace(0, w - 1, 40), np.linspace(0, a_.shape[0] - 1, 10))
            g = np.float32(np.c_[gx.ravel(), gy.ravel()])
            tg = transfer(Hc, g); ok = (tg[:, 0] >= 0) & (tg[:, 0] < w) & (tg[:, 1] >= 0) & (tg[:, 1] < a_.shape[0])
            if ok.sum():
                err = float(np.linalg.norm(transfer(He, g[ok]) - tg[ok], axis=1).mean())
        label = "кадр цілком" if y0f == 0 else f"нижні {round(100 * (1 - y0f))} % кадру"
        rows2.append((label, a_.shape[0], ncand, ninl, 100 * ninl / ncand if ncand else 0.0, err))
        print(f"E2 {label}: кандидатів {ncand}, узгоджених {ninl}, похибка {err:.2f} px")
        if vis and y0f == 0.72:
            k1, k2, m, mk = vis
            bad = [mm for mm, ok_ in zip(m, mk) if not ok_][:60]; good = [mm for mm, ok_ in zip(m, mk) if ok_][:60]
            im = cv2.drawMatches(a_, k1, b_, k2, bad, None, matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            im = cv2.drawMatches(a_, k1, b_, k2, good, im, matchColor=(0, 200, 0), flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(os.path.join(FIG, "e2_pavement.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open(os.path.join(RES, "e2_periodic.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["region", "height_px", "candidates", "inliers", "inlier_share_pct", "err_vs_ref_px"])
        for r in rows2: wr.writerow([r[0], r[1], r[2], r[3], round(r[4], 1), round(r[5], 2)])

    # ---------- E3: ланцюжки з 3, 4, 5 кадрів (перші k кадрів з data/frame_*.jpg)
    frames, _ = load_frames()
    sets = {f"{k} кадри": frames[:k] for k in range(3, len(frames) + 1)}
    rows3 = []
    for i, (name, fr) in enumerate(sets.items()):
        r = part_d.run(fr, tag=f"e3_{i}")
        rows3.append((name, r["n"], r["canvas"][0], r["canvas"][1], r["area_ratio"], r["mean_err"], r["inliers"]))
        print(f"E3 {name}: полотно {r['canvas'][0]}x{r['canvas'][1]}, відношення площ {r['area_ratio']:.1f}, похибка {r['mean_err']:.2f}")
    with open(os.path.join(RES, "e3_chains.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["set", "frames", "canvas_w", "canvas_h", "area_ratio", "mean_err_px", "total_inliers"])
        for r in rows3: wr.writerow([r[0], r[1], r[2], r[3], round(r[4], 1), round(r[5], 3), r[6]])


def run_keyboard():
    """E4: два кадри клавіатури майже без спільної частини (data/fail_ov5_1.jpg, fail_ov5_2.jpg): однакові темні клавіші дають хибні пари."""
    import common
    p1, p2 = os.path.join(DATA, "fail_ov5_1.jpg"), os.path.join(DATA, "fail_ov5_2.jpg")
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print("E4: fail_ov5_1.jpg / fail_ov5_2.jpg не знайдено, пропускаю")
        return
    a, b = common._read(p1), common._read(p2)
    k1, d1 = sift_features(a); k2, d2 = sift_features(b)
    knn = knn_pairs(d1, d2)
    rows = []
    for T in (0.75, 0.9):
        m = mutual_filter(ratio_filter(knn, T), d1, d2)
        ninl = 0
        if len(m) >= 4:
            src, dst = pts_from_matches(m, k1, k2)
            H, mask = ransac_homography(src, dst, 3.0, 2000, seed=0)
            ninl = int(mask.sum())
            if T == 0.9:
                bad = [mm for mm, ok in zip(m, mask) if not ok][:60]; good = [mm for mm, ok in zip(m, mask) if ok][:60]
                im = cv2.drawMatches(a, k1, b, k2, bad, None, matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                im = cv2.drawMatches(a, k1, b, k2, good, im, matchColor=(0, 200, 0), flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(os.path.join(FIG, "e4_keyboard.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append((T, len(k1), len(k2), len(m), ninl))
        print(f"E4 клавіатура T={T}: точок {len(k1)}/{len(k2)}, кандидатів {len(m)}, узгоджених {ninl}")
    with open(os.path.join(RES, "e4_keyboard.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["T", "kp1", "kp2", "candidates_mutual", "inliers"])
        for r in rows: wr.writerow(r)


def run_stability(n_seeds=8):
    """E3b: стабільність панорами — той самий набір кадрів, різні випадкові вибірки RANSAC (SIFT і зіставлення рахуються один раз)."""
    frames, _ = load_frames()
    sets = {f"{k} кадри": frames[:k] for k in sorted({3, len(frames)}) if k <= len(frames)}
    rows = []
    for name, fr in sets.items():
        n = len(fr); M = part_d.pairwise_inliers(fr)
        edges = part_d.max_spanning_tree(M); ref, adj = part_d.tree_center(n, edges)
        feats = [sift_features(f) for f in fr]
        order, q, par = [], [ref], {ref: None}
        for u in q:
            for v in adj[u]:
                if v not in par: par[v] = u; q.append(v); order.append(v)
        data = {}
        for v in order:
            u = par[v]; k1, d1 = feats[v]; k2, d2 = feats[u]
            m = mutual_filter(ratio_filter(knn_pairs(d1, d2), 0.75), d1, d2)
            data[v] = pts_from_matches(m, k1, k2)
        h0, w0 = fr[0].shape[:2]; ratios = []
        for sd in range(n_seeds):
            T = {ref: np.eye(3)}
            for v in order:
                H, _ = ransac_homography(*data[v], 3.0, 2000, seed=1000 * sd + v)
                T[v] = T[par[v]] @ H
            cs = np.vstack([transfer(T[i], np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]])) for i in range(n)])
            Wc, Hc = cs[:, 0].max() - cs[:, 0].min(), cs[:, 1].max() - cs[:, 1].min()
            ratios.append(Wc * Hc / (n * w0 * h0))
        rows.append((name, min(ratios), float(np.median(ratios)), max(ratios)))
        print(f"E3b {name}: відношення площ мін {min(ratios):.1f}, медіана {np.median(ratios):.1f}, макс {max(ratios):.1f}")
    with open(os.path.join(RES, "e3b_stability.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["set", "area_ratio_min", "area_ratio_median", "area_ratio_max"])
        for r in rows: wr.writerow([r[0], round(r[1], 1), round(r[2], 1), round(r[3], 1)])


if __name__ == "__main__":
    run(); run_keyboard(); run_stability()