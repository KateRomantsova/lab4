"""Частина C. Власний RANSAC (кроки 8-12)."""
import csv, os
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from part_a import get_candidates

TAU = 3.0
N_SET = 2000
TAUS = [1, 2, 3, 5, 8, 12, 20]


def run():
    img1, img2, k1, k2, d1, d2, m = get_candidates()
    src, dst = pts_from_matches(m, k1, k2)
    n = len(src)

    # крок 8: власний RANSAC
    H, mask = ransac_homography(src, dst, TAU, N_SET, seed=0)
    err_in = reproj_errors(H, src, dst)[mask]
    eps = 1 - mask.sum() / n
    # крок 9: потрібна кількість ітерацій
    N_req = required_iterations(eps, 0.99, 4)
    print(f"власний RANSAC: узгоджених {mask.sum()}/{n}, частка викидів {eps:.3f}, N_потрібне = {N_req:.1f}, задано {N_SET}; похибка {err_in.mean():.3f} px")

    # крок 10: звірка з cv2.RANSAC
    Hc, mc = cv2.findHomography(src, dst, cv2.RANSAC, TAU)
    mc = mc.ravel().astype(bool)
    err_cv = reproj_errors(Hc, src, dst)[mc]
    common_in = int((mask & mc).sum())
    print(f"cv2.RANSAC: узгоджених {mc.sum()}, похибка {err_cv.mean():.3f} px; спільних {common_in}")
    with open(os.path.join(RES, "c_compare.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "inliers", "mean_err_px"])
        w.writerow(["own_RANSAC", int(mask.sum()), round(err_in.mean(), 3)])
        w.writerow(["cv2_RANSAC", int(mc.sum()), round(err_cv.mean(), 3)])
    with open(os.path.join(RES, "c_iterations.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pairs", "inliers", "eps", "N_required", "N_set"])
        w.writerow([n, int(mask.sum()), round(eps, 3), round(N_req, 1), N_SET])

    # крок 11: дослідження порога
    rows = []
    for tau in TAUS:
        Ht, mt = ransac_homography(src, dst, tau, N_SET, seed=0)
        e = reproj_errors(Ht, src, dst)[mt].mean()
        Hc2, mc2 = cv2.findHomography(src, dst, cv2.RANSAC, float(tau))
        mc2 = mc2.ravel().astype(bool)
        e2 = reproj_errors(Hc2, src, dst)[mc2].mean()
        rows.append((tau, int(mt.sum()), e, int(mc2.sum()), e2))
        print(f"τ={tau}: власний {mt.sum()} / {e:.3f} px; cv2 {mc2.sum()} / {e2:.3f} px")
    with open(os.path.join(RES, "c_threshold.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tau", "own_inliers", "own_err", "cv2_inliers", "cv2_err"])
        for r in rows: w.writerow([r[0], r[1], round(r[2], 3), r[3], round(r[4], 3)])
    r = np.array(rows, dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].plot(r[:, 0], r[:, 1], "o-", label="власний RANSAC"); ax[0].plot(r[:, 0], r[:, 3], "s--", label="cv2.RANSAC")
    ax[0].set_xlabel("поріг τ, пікселів"); ax[0].set_ylabel("кількість узгоджених точок"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[0].set_ylim(0, 1.15 * r[:, 1].max()); ax[0].set_title("Узгоджені точки від τ")
    ax[1].plot(r[:, 0], r[:, 2], "o-", label="власний RANSAC"); ax[1].plot(r[:, 0], r[:, 4], "s--", label="cv2.RANSAC")
    ax[1].set_xlabel("поріг τ, пікселів"); ax[1].set_ylabel("середня похибка репроєкції, px"); ax[1].grid(alpha=.3); ax[1].legend()
    ax[1].set_title("Похибка на узгоджених від τ")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "c_threshold.png"), dpi=170); plt.close()

    # крок 12: узгоджені (зелені) й відкинуті (червоні) зіставлення
    bad = [mm for mm, ok in zip(m, mask) if not ok]
    good = [mm for mm, ok in zip(m, mask) if ok]
    rng = np.random.default_rng(1)
    bad_s = [bad[i] for i in rng.choice(len(bad), min(80, len(bad)), replace=False)]
    good_s = [good[i] for i in rng.choice(len(good), min(80, len(good)), replace=False)]
    vis = cv2.drawMatches(img1, k1, img2, k2, bad_s, None, matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis = cv2.drawMatches(img1, k1, img2, k2, good_s, vis, matchColor=(0, 200, 0),
                          flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(FIG, "c_inliers_outliers.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    run()
