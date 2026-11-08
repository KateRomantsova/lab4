"""Спільні функції лабораторної: дані, SIFT, зіставлення, RANSAC, похибка репроєкції."""
import os
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("LAB_DATA", os.path.join(ROOT, "data"))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")


# ------------------------------------------------------------------ дані
WIDTH = 1000      # усі кадри зменшуються до цієї ширини (швидкість; координати в звіті — у цих пікселях)


def _read(path):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    s = WIDTH / im.shape[1]
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def load_pair():
    """Пара сусідніх кадрів для частин A-C: data/pair_1.jpg, data/pair_2.jpg (перекриття ≈ 40 %)."""
    return _read(os.path.join(DATA, "pair_1.jpg")), _read(os.path.join(DATA, "pair_2.jpg"))


def load_frames():
    """Кадри панорами data/frame_1.jpg ... frame_N.jpg (порядок у файлах не обов'язково просторовий)."""
    files = sorted(f for f in os.listdir(DATA) if f.startswith("frame_"))
    return [_read(os.path.join(DATA, f)) for f in files], files


# ------------------------------------------------------------ ознаки
def sift_features(img, nfeatures=0):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(g, None)          # (keypoints, descriptors)


def knn_pairs(d1, d2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    return bf.knnMatch(d1, d2, k=2)


def ratio_filter(knn, T):
    """Критерій відношення відстаней Лоу: d1 < T * d2."""
    return [m for m, n in knn if m.distance < T * n.distance]


def mutual_filter(matches, d1, d2):
    """Взаємна перевірка: найближчий сусід точки з кадру 2 в кадрі 1 має збігатися."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    back = {m.queryIdx: m.trainIdx for m in bf.match(d2, d1)}   # idx в кадрі 2 -> idx в кадрі 1
    return [m for m in matches if back.get(m.trainIdx) == m.queryIdx]


def pts_from_matches(matches, k1, k2):
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return src, dst


# ------------------------------------------------- гомографія й похибка
def transfer(H, pts):
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


def reproj_errors(H, src, dst):
    return np.linalg.norm(transfer(H, src) - dst, axis=1)


def required_iterations(eps, p=0.99, s=4):
    """N = log(1 - p) / log(1 - (1 - eps)^s)."""
    good = (1.0 - eps) ** s
    if good <= 0:
        return float("inf")
    return np.log(1.0 - p) / np.log(1.0 - good)


def ransac_homography(src, dst, tau=3.0, n_iter=2000, seed=0):
    """Власний RANSAC: 4 випадкові пари -> getPerspectiveTransform -> підрахунок узгоджених ->
    переоцінка за всіма узгодженими через findHomography(..., 0)."""
    rng = np.random.default_rng(seed)
    n = len(src)
    best_cnt, best_mask = -1, None
    for _ in range(n_iter):
        idx = rng.choice(n, 4, replace=False)
        try:
            H = cv2.getPerspectiveTransform(src[idx], dst[idx])
        except cv2.error:
            continue
        if H is None or not np.all(np.isfinite(H)):
            continue
        mask = reproj_errors(H, src, dst) < tau
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_cnt, best_mask = cnt, mask
    if best_mask is None or best_cnt < 4:
        return None, np.zeros(n, bool)
    H, _ = cv2.findHomography(src[best_mask], dst[best_mask], 0)
    return H, best_mask


def pair_homography(img1, img2, T=0.75, tau=3.0, n_iter=2000, seed=0):
    """Повний конвеєр для пари кадрів: SIFT -> ratio -> mutual -> власний RANSAC."""
    k1, d1 = sift_features(img1); k2, d2 = sift_features(img2)
    m = mutual_filter(ratio_filter(knn_pairs(d1, d2), T), d1, d2)
    src, dst = pts_from_matches(m, k1, k2)
    H, mask = ransac_homography(src, dst, tau, n_iter, seed)
    return H, src, dst, mask
