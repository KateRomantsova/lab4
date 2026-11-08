"""Частина A. Зіставлення ознак (кроки 1-5)."""
import csv, os
import numpy as np, cv2
from common import *

T_LIST = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
T_RAW = 0.9       # поріг для «сирих» кандидатів, які далі йдуть у частини B і C


def get_pair():
    return load_pair()


def get_candidates(T=T_RAW):
    img1, img2 = get_pair()
    k1, d1 = sift_features(img1); k2, d2 = sift_features(img2)
    m = ratio_filter(knn_pairs(d1, d2), T)
    return img1, img2, k1, k2, d1, d2, m


def run():
    img1, img2 = get_pair()
    k1, d1 = sift_features(img1); k2, d2 = sift_features(img2)                 # крок 1
    knn = knn_pairs(d1, d2)                                                    # крок 2
    rows = []
    for T in T_LIST:                                                           # кроки 3-4
        m = ratio_filter(knn, T)
        mm = mutual_filter(m, d1, d2)
        def n_inl(mt):
            if len(mt) < 4:
                return 0
            a, b = pts_from_matches(mt, k1, k2)
            _, mk = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
            return int(mk.sum())
        rows.append((T, len(m), n_inl(m), len(mm), n_inl(mm)))
        print(f"T={T}: кандидатів {len(m)} (узгоджених {rows[-1][2]}), після взаємної перевірки {len(mm)} (узгоджених {rows[-1][4]})")
    with open(os.path.join(RES, "a_ratio.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["T", "candidates", "inliers", "mutual", "inliers_mutual"]); w.writerows(rows)
    with open(os.path.join(RES, "a_info.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kp1", "kp2", "desc_dim"]); w.writerow([len(k1), len(k2), d1.shape[1]])

    # крок 5: візуалізація сирих кандидатів (T = 0.9), підвибірка для читабельності
    m = ratio_filter(knn, T_RAW)
    rng = np.random.default_rng(0)
    sub = [m[i] for i in rng.choice(len(m), min(120, len(m)), replace=False)]
    vis = cv2.drawMatches(img1, k1, img2, k2, sub, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(FIG, "a_raw_matches.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    run()
