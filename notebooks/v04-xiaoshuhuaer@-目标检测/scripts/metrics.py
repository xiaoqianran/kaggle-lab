"""Detection metrics: IoU family, NMS, COCO-style AP (single-process pure numpy)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from boxes import box_area_xyxy

Array = np.ndarray


def box_iou_xyxy(boxes1: Array, boxes2: Array) -> Array:
    """Pairwise IoU matrix [N, M] for xyxy boxes."""
    b1 = np.asarray(boxes1, dtype=np.float64)
    b2 = np.asarray(boxes2, dtype=np.float64)
    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]
    area1 = box_area_xyxy(b1)
    area2 = box_area_xyxy(b2)
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def box_giou_xyxy(boxes1: Array, boxes2: Array) -> Array:
    """Pairwise GIoU [N, M]."""
    b1 = np.asarray(boxes1, dtype=np.float64)
    b2 = np.asarray(boxes2, dtype=np.float64)
    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]
    iou = box_iou_xyxy(b1, b2)
    area1 = box_area_xyxy(b1)
    area2 = box_area_xyxy(b2)
    lt = np.minimum(b1[:, None, :2], b2[None, :, :2])
    rb = np.maximum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    enclose = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - (
        iou * np.maximum(area1[:, None] + area2[None, :] - iou * 0 + 0, 1e-12)
    )
    # recompute union properly
    inter_lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    inter_rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    inter_wh = np.maximum(0.0, inter_rb - inter_lt)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return iou - (enclose - union) / np.maximum(enclose, 1e-12)


def nms_xyxy(
    boxes: Array, scores: Array, iou_thr: float = 0.5, max_det: int = 100
) -> Array:
    """Greedy NMS. Returns indices into the original arrays."""
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = box_iou_xyxy(boxes[i], boxes[rest]).reshape(-1)
        order = rest[ious <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def soft_nms_xyxy(
    boxes: Array,
    scores: Array,
    iou_thr: float = 0.5,
    sigma: float = 0.5,
    score_thr: float = 0.001,
    max_det: int = 100,
) -> Tuple[Array, Array]:
    """Linear Soft-NMS variant (Gaussian decay). Returns kept indices and updated scores."""
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    scores = np.asarray(scores, dtype=np.float64).copy()
    N = len(boxes)
    if N == 0:
        return np.zeros((0,), dtype=np.int64), scores
    idxs = np.arange(N)
    keep_idx: List[int] = []
    keep_scores: List[float] = []
    while idxs.size > 0 and len(keep_idx) < max_det:
        m = int(np.argmax(scores[idxs]))
        i = int(idxs[m])
        if scores[i] < score_thr:
            break
        keep_idx.append(i)
        keep_scores.append(float(scores[i]))
        idxs = np.delete(idxs, m)
        if idxs.size == 0:
            break
        ious = box_iou_xyxy(boxes[i], boxes[idxs]).reshape(-1)
        decay = np.exp(-(ious**2) / sigma)
        # only decay highly overlapping (optional soft mask)
        scores[idxs] = scores[idxs] * np.where(ious > iou_thr, decay, 1.0)
    return np.asarray(keep_idx, dtype=np.int64), np.asarray(keep_scores, dtype=np.float64)


@dataclass
class APResult:
    ap50: float
    ap: float  # mean over IoU thr 0.5:0.95
    ap_s: float
    ap_m: float
    ap_l: float
    ar100: float
    n_gt: int
    n_pred: int

    def as_dict(self) -> Dict:
        return asdict(self)


def _box_size_label(boxes: Array) -> Array:
    """COCO-like area bins: S <32^2, M <96^2, else L. boxes xyxy."""
    areas = box_area_xyxy(boxes)
    labels = np.full(len(boxes), "l", dtype=object)
    labels[areas < 32**2] = "s"
    labels[(areas >= 32**2) & (areas < 96**2)] = "m"
    return labels


def average_precision_pr(recalls: Array, precisions: Array) -> float:
    """VOC/COCO-style AP via precision envelope + trapezoid on recall."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_image_class(
    pred_boxes: Array,
    pred_scores: Array,
    gt_boxes: Array,
    iou_thr: float,
) -> Tuple[Array, Array, int]:
    """Match preds to gts for one image/class. Returns tp, fp arrays sorted by score desc, n_gt."""
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    n_gt = len(gt_boxes)
    if len(pred_boxes) == 0:
        return np.zeros(0), np.zeros(0), n_gt
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched = np.zeros(n_gt, dtype=bool)
    if n_gt == 0:
        fp[:] = 1
        return tp, fp, n_gt
    ious = box_iou_xyxy(pred_boxes, gt_boxes)
    for i in range(len(pred_boxes)):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and not matched[j]:
            tp[i] = 1
            matched[j] = True
        else:
            fp[i] = 1
    return tp, fp, n_gt


def coco_style_ap(
    predictions: Sequence[Dict],
    ground_truths: Sequence[Dict],
    iou_thrs: Optional[Sequence[float]] = None,
    conf_thr: float = 0.0,
) -> APResult:
    """
    predictions / ground_truths: list of dicts per image
      pred: {"boxes": Nx4 xyxy, "scores": N, "labels": N int}
      gt:   {"boxes": Mx4, "labels": M}
    Multi-class micro-style: evaluate per class then mean.
    """
    if iou_thrs is None:
        iou_thrs = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    # collect classes
    classes = set()
    for g in ground_truths:
        if len(g.get("labels", [])):
            classes.update(np.asarray(g["labels"]).tolist())
    for p in predictions:
        if len(p.get("labels", [])):
            classes.update(np.asarray(p["labels"]).tolist())
    if not classes:
        return APResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    classes = sorted(classes)

    def ap_at_iou(iou_thr: float, size_filter: Optional[str] = None) -> float:
        aps = []
        for c in classes:
            tps_all = []
            fps_all = []
            n_gt = 0
            scores_all = []
            for pred, gt in zip(predictions, ground_truths):
                pb = np.asarray(pred.get("boxes", np.zeros((0, 4))), dtype=np.float64)
                ps = np.asarray(pred.get("scores", np.zeros(0)), dtype=np.float64)
                pl = np.asarray(pred.get("labels", np.zeros(0)), dtype=np.int64)
                gb = np.asarray(gt.get("boxes", np.zeros((0, 4))), dtype=np.float64)
                gl = np.asarray(gt.get("labels", np.zeros(0)), dtype=np.int64)
                if len(pb):
                    m = (pl == c) & (ps >= conf_thr)
                    pb, ps = pb[m], ps[m]
                else:
                    pb, ps = pb, ps
                if len(gb):
                    gm = gl == c
                    gb = gb[gm]
                if size_filter is not None and len(gb):
                    sizes = _box_size_label(gb)
                    gb = gb[sizes == size_filter]
                if size_filter is not None and len(pb):
                    sizes = _box_size_label(pb)
                    # size filter on GT only for AP_s/m/l (COCO); keep preds
                tp, fp, ng = evaluate_image_class(pb, ps, gb, iou_thr)
                n_gt += ng
                if len(tp):
                    tps_all.append(tp)
                    fps_all.append(fp)
                    scores_all.append(ps)
            if n_gt == 0:
                continue
            if not tps_all:
                aps.append(0.0)
                continue
            scores = np.concatenate(scores_all)
            tp = np.concatenate(tps_all)
            fp = np.concatenate(fps_all)
            order = np.argsort(-scores)
            tp = np.cumsum(tp[order])
            fp = np.cumsum(fp[order])
            rec = tp / n_gt
            prec = tp / np.maximum(tp + fp, 1e-12)
            aps.append(average_precision_pr(rec, prec))
        return float(np.mean(aps)) if aps else 0.0

    ap50 = ap_at_iou(0.5)
    aps = [ap_at_iou(t) for t in iou_thrs]
    ap = float(np.mean(aps))
    ap_s = ap_at_iou(0.5, "s")
    ap_m = ap_at_iou(0.5, "m")
    ap_l = ap_at_iou(0.5, "l")

    # AR@100 simple: max recall at iou=0.5 with top 100 preds/image
    recalls = []
    n_gt_total = 0
    n_pred_total = 0
    for pred, gt in zip(predictions, ground_truths):
        gb = np.asarray(gt.get("boxes", np.zeros((0, 4))), dtype=np.float64)
        n_gt_total += len(gb)
        pb = np.asarray(pred.get("boxes", np.zeros((0, 4))), dtype=np.float64)
        ps = np.asarray(pred.get("scores", np.zeros(0)), dtype=np.float64)
        n_pred_total += len(pb)
        if len(gb) == 0:
            continue
        if len(pb) == 0:
            recalls.append(0.0)
            continue
        order = np.argsort(-ps)[:100]
        pb = pb[order]
        ious = box_iou_xyxy(pb, gb)
        matched = 0
        used = np.zeros(len(gb), dtype=bool)
        for i in range(len(pb)):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= 0.5 and not used[j]:
                used[j] = True
                matched += 1
        recalls.append(matched / len(gb))
    ar100 = float(np.mean(recalls)) if recalls else 0.0

    return APResult(
        ap50=ap50,
        ap=ap,
        ap_s=ap_s,
        ap_m=ap_m,
        ap_l=ap_l,
        ar100=ar100,
        n_gt=n_gt_total,
        n_pred=n_pred_total,
    )


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
