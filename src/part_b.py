"""Частина B. Чому потрібен робастний метод (кроки 6-7)."""
import csv, os
import numpy as np, cv2
from common import *
from part_a import get_candidates


def run():
    img1, img2, k1, k2, d1, d2, m = get_candidates()
    src, dst = pts_from_matches(m, k1, k2)
    H, _ = cv2.findHomography(src, dst, 0)                         # крок 6: МНК без відсіву
    err = reproj_errors(H, src, dst)
    print(f"МНК: {len(src)} пар, середня похибка {err.mean():.1f} px, медіана {np.median(err):.1f} px")
    with open(os.path.join(RES, "b_lsq.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pairs", "mean_err", "median_err"]); w.writerow([len(src), round(err.mean(), 2), round(np.median(err), 2)])
    # крок 7: warpPerspective першого кадру у систему другого; полотно фіксоване: [-W, 2W] x [-H/2, 3H/2]
    h, w = img2.shape[:2]
    shift = np.array([[1, 0, w], [0, 1, h // 2], [0, 0, 1]], np.float64)
    size = (3 * w, 2 * h)
    warped = cv2.warpPerspective(img1, shift @ H, size)
    base = cv2.warpPerspective(img2, shift, size)
    cv2.imwrite(os.path.join(FIG, "b_lsq_warp.jpg"), cv2.resize(warped, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    over = cv2.addWeighted(warped, 0.55, base, 0.45, 0)
    cv2.imwrite(os.path.join(FIG, "b_lsq_overlay.jpg"), cv2.resize(over, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    run()
