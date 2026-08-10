"""Label assignment: IoU threshold matching and Hungarian (DETR-style)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxes import xyxy_to_cxcywh
from metrics import box_iou_xyxy

Array = np.ndarray


def assign_iou(
    anchors_xyxy: Array,
    gt_xyxy: Array,
    pos_thr: float = 0.7,
    neg_thr: float = 0.3,
) -> Dict[str, Array]:
    """
    Faster-RCNN style: positive if IoU>=pos_thr or best-matching anchor per GT;
    negative if IoU<neg_thr; ignore otherwise (-1).
    Returns labels in {-1,0,1}, matched_gt index (-1 if none).
    """
    anchors = np.asarray(anchors_xyxy, dtype=np.float64)
    gts = np.asarray(gt_xyxy, dtype=np.float64)
    n_a = len(anchors)
    labels = np.full(n_a, -1, dtype=np.int64)
    matched_gt = np.full(n_a, -1, dtype=np.int64)
    if n_a == 0:
        return {"labels": labels, "matched_gt": matched_gt, "max_iou": np.zeros(0)}
    if len(gts) == 0:
        labels[:] = 0
        return {"labels": labels, "matched_gt": matched_gt, "max_iou": np.zeros(n_a)}

    ious = box_iou_xyxy(anchors, gts)  # [A,G]
    max_iou = ious.max(axis=1)
    argmax_gt = ious.argmax(axis=1)
    labels[max_iou < neg_thr] = 0
    labels[max_iou >= pos_thr] = 1
    matched_gt[max_iou >= pos_thr] = argmax_gt[max_iou >= pos_thr]

    # ensure each GT has at least one positive (best anchor)
    best_anchor_for_gt = ious.argmax(axis=0)
    for g, a in enumerate(best_anchor_for_gt):
        labels[a] = 1
        matched_gt[a] = g
    return {"labels": labels, "matched_gt": matched_gt, "max_iou": max_iou, "ious": ious}


def hungarian_match(
    pred_boxes_xyxy: Array,
    pred_logits: Array,
    gt_boxes_xyxy: Array,
    gt_labels: Array,
    num_classes: int,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> Dict[str, Array]:
    """
    DETR-style bipartite matching.
    pred_logits: [Q, C] class logits (no background required; last class can be no-object).
    gt_labels: [G] in 0..C-2 if background is last, or 0..C-1.
    Cost = cost_class * (-prob[gt_label]) + cost_bbox * L1(cxcywh) + cost_giou * (1-giou)
    """
    from metrics import box_giou_xyxy

    pb = np.asarray(pred_boxes_xyxy, dtype=np.float64)
    logits = np.asarray(pred_logits, dtype=np.float64)
    gb = np.asarray(gt_boxes_xyxy, dtype=np.float64)
    gl = np.asarray(gt_labels, dtype=np.int64)
    Q = len(pb)
    G = len(gb)
    if G == 0 or Q == 0:
        return {
            "pred_idx": np.zeros(0, dtype=np.int64),
            "gt_idx": np.zeros(0, dtype=np.int64),
            "cost": np.zeros((Q, max(G, 1))),
        }

    # softmax probs
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = e / e.sum(axis=1, keepdims=True)

    pc = xyxy_to_cxcywh(pb)
    gc = xyxy_to_cxcywh(gb)
    # L1 cost [Q,G]
    l1 = np.abs(pc[:, None, :] - gc[None, :, :]).sum(axis=-1)
    giou = box_giou_xyxy(pb, gb)  # [Q,G]
    # class cost: -prob of target class
    class_cost = np.zeros((Q, G))
    for j, lab in enumerate(gl):
        class_cost[:, j] = -prob[:, int(lab)]

    cost = cost_class * class_cost + cost_bbox * l1 + cost_giou * (1.0 - giou)
    row_ind, col_ind = linear_sum_assignment(cost)
    return {
        "pred_idx": row_ind.astype(np.int64),
        "gt_idx": col_ind.astype(np.int64),
        "cost": cost,
        "matched_cost": cost[row_ind, col_ind],
    }


def simota_like_assign(
    pred_boxes: Array,
    pred_scores: Array,
    gt_boxes: Array,
    topk: int = 10,
    center_radius: float = 2.5,
    stride: float = 8.0,
) -> Dict[str, Array]:
    """
    Simplified SimOTA-style: for each GT pick top-k predictions by cost=cls*iou + center prior.
    pred_scores: [N] objectness or class score for single-class case.
    Returns positive mask over predictions and matched gt index.
    """
    pb = np.asarray(pred_boxes, dtype=np.float64)
    ps = np.asarray(pred_scores, dtype=np.float64)
    gb = np.asarray(gt_boxes, dtype=np.float64)
    N, G = len(pb), len(gb)
    pos = np.zeros(N, dtype=bool)
    matched = np.full(N, -1, dtype=np.int64)
    if N == 0 or G == 0:
        return {"pos": pos, "matched_gt": matched}

    ious = box_iou_xyxy(pb, gb)
    # center of preds and gts
    pcx = 0.5 * (pb[:, 0] + pb[:, 2])
    pcy = 0.5 * (pb[:, 1] + pb[:, 3])
    gcx = 0.5 * (gb[:, 0] + gb[:, 2])
    gcy = 0.5 * (gb[:, 1] + gb[:, 3])
    for g in range(G):
        # center prior: within radius*stride of GT center
        dist = np.sqrt((pcx - gcx[g]) ** 2 + (pcy - gcy[g]) ** 2)
        prior = dist < (center_radius * stride)
        # cost lower is better
        cost = -np.log(np.clip(ps, 1e-6, 1.0)) - np.log(np.clip(ious[:, g], 1e-6, 1.0))
        cost = np.where(prior, cost, cost + 10.0)
        k = min(topk, N)
        idx = np.argpartition(cost, k - 1)[:k]
        pos[idx] = True
        # assign if better than existing (by iou)
        for i in idx:
            if matched[i] < 0 or ious[i, g] > ious[i, matched[i]]:
                matched[i] = g
    return {"pos": pos, "matched_gt": matched, "ious": ious}
