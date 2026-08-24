from __future__ import annotations
import numpy as np

AREA_BUCKETS = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, np.inf)}


def iou_matrix(preds, gts):
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)), dtype=np.float32)
    p, g = preds[:, None, :], gts[None, :, :]
    x1 = np.maximum(p[..., 0], g[..., 0]); y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2]); y2 = np.minimum(p[..., 3], g[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    ag = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    union = ap[:, None] + ag[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match_image(preds, scores, gts, iou_thr):
    """Greedy match, predictions taken in descending-confidence order."""
    order = np.argsort(-scores)
    preds, scores = preds[order], scores[order]
    ious = iou_matrix(preds, gts)
    tp = np.zeros(len(preds), dtype=bool)
    matched = np.full(len(preds), -1, dtype=int)
    taken = set()
    for i in range(len(preds)):
        best_j, best_iou = -1, iou_thr
        for j in range(len(gts)):
            if j in taken:
                continue
            if ious[i, j] >= best_iou:
                best_iou, best_j = ious[i, j], j
        if best_j >= 0:
            tp[i] = True; matched[i] = best_j; taken.add(best_j)
    return tp, matched, scores


def found_set(preds, scores, gts, iou_thr=0.5, conf=0.25):
    """Indices of ground-truth boxes this detector found at the given threshold."""
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    gts = np.asarray(gts, dtype=np.float64).reshape(-1, 4)
    keep = scores >= conf
    tp, matched, _ = match_image(preds[keep], scores[keep], gts, iou_thr)
    return {int(m) for m, t in zip(matched, tp) if t}


def average_precision(tp, scores, n_gt):
    """101-point interpolated AP (COCO)."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order]
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    return float(np.where(idx < len(precision),
                          precision[np.minimum(idx, len(precision) - 1)], 0.0).mean())


def _clean(dataset):
    return [{"preds":  np.asarray(d["preds"],  dtype=np.float64).reshape(-1, 4),
             "scores": np.asarray(d["scores"], dtype=np.float64).reshape(-1),
             "gts":    np.asarray(d["gts"],    dtype=np.float64).reshape(-1, 4)}
            for d in dataset]


def confidence_sweep(dataset, iou_thr=0.5, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    n_gt = sum(len(d["gts"]) for d in dataset)
    cache = [match_image(d["preds"], d["scores"], d["gts"], iou_thr) for d in dataset]
    rows = []
    for t in thresholds:
        tp = fp = 0
        for flags, _, sc in cache:
            keep = sc >= t
            tp += int(flags[keep].sum()); fp += int((~flags[keep]).sum())
        p = tp / max(tp + fp, 1e-9); r = tp / max(n_gt, 1e-9)
        rows.append({"conf": float(t), "tp": tp, "fp": fp, "fn": n_gt - tp,
                     "precision": p, "recall": r,
                     "f1": 2 * p * r / max(p + r, 1e-9)})
    return rows


def recall_by_area(dataset, iou_thr=0.5, conf=0.25):
    hits = {k: 0 for k in AREA_BUCKETS}; totals = {k: 0 for k in AREA_BUCKETS}
    for d in dataset:
        found = found_set(d["preds"], d["scores"], d["gts"], iou_thr, conf)
        for j, gt in enumerate(d["gts"]):
            area = (gt[2] - gt[0]) * (gt[3] - gt[1])
            for name, (lo, hi) in AREA_BUCKETS.items():
                if lo <= area < hi:
                    totals[name] += 1
                    if j in found:
                        hits[name] += 1
                    break
    return {k: {"recall": hits[k] / totals[k] if totals[k] else float("nan"),
                "n": totals[k]} for k in AREA_BUCKETS}


def evaluate(dataset, iou_thrs=None, area_conf=0.25):
    if iou_thrs is None:
        iou_thrs = np.arange(0.5, 1.0, 0.05)
    dataset = _clean(dataset)
    n_gt = sum(len(d["gts"]) for d in dataset)
    aps = {}
    for thr in iou_thrs:
        all_tp, all_sc = [], []
        for d in dataset:
            tp, _, sc = match_image(d["preds"], d["scores"], d["gts"], thr)
            all_tp.append(tp); all_sc.append(sc)
        tp = np.concatenate(all_tp) if all_tp else np.array([], dtype=bool)
        sc = np.concatenate(all_sc) if all_sc else np.array([])
        aps[round(float(thr), 2)] = average_precision(tp, sc, n_gt)
    sweep = confidence_sweep(dataset, 0.5)
    best = max(sweep, key=lambda r: r["f1"])
    return {"AP50": aps[0.5], "AP50_95": float(np.nanmean(list(aps.values()))),
            "AP_by_iou": aps, "n_gt": n_gt, "best_f1": best, "sweep": sweep,
            "recall_by_area": recall_by_area(dataset, 0.5, area_conf)}


def yolo_txt_to_xyxy(path, img_w, img_h, with_conf=False):
    boxes, scores = [], []
    try:
        lines = open(path).read().strip().splitlines()
    except FileNotFoundError:
        return np.zeros((0, 4)), np.zeros(0)
    for line in lines:
        v = line.split()
        if len(v) < 5:
            continue
        xc, yc, w, h = (float(x) for x in v[1:5])
        boxes.append([(xc - w/2)*img_w, (yc - h/2)*img_h,
                      (xc + w/2)*img_w, (yc + h/2)*img_h])
        scores.append(float(v[5]) if (with_conf and len(v) >= 6) else 1.0)
    return np.array(boxes).reshape(-1, 4), np.array(scores)
