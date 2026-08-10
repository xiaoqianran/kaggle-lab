"""Minimal trainable detectors for P2–P5 (CPU-friendly)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyBackbone(nn.Module):
    """64x64 -> 8x8 feature map, C=64."""

    def __init__(self, c=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(3, 32, 3, 2, 1),  # 32
            ConvBNReLU(32, 32),
            ConvBNReLU(32, 64, 3, 2, 1),  # 16
            ConvBNReLU(64, 64),
            ConvBNReLU(64, c, 3, 2, 1),  # 8
            ConvBNReLU(c, c),
        )

    def forward(self, x):
        return self.net(x)


class CenterNetLite(nn.Module):
    """
    Single-stage center-based detector (FCOS/YOLO-lite flavor):
    predict per-cell: objectness, class logits, and ltrb distances to box sides.
    Grid is 8x8 on 64x64 images (stride=8).
    """

    def __init__(self, n_classes: int = 3, stride: int = 8, img_size: int = 64):
        super().__init__()
        self.n_classes = n_classes
        self.stride = stride
        self.img_size = img_size
        self.backbone = TinyBackbone(64)
        self.head = nn.Conv2d(64, 1 + n_classes + 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 1+C+4, H, W]
        return self.head(self.backbone(x))


def decode_centernet(
    out: torch.Tensor,
    stride: int = 8,
    conf_thr: float = 0.3,
    n_classes: int = 3,
) -> List[Dict[str, torch.Tensor]]:
    """Decode batch of head outputs to boxes xyxy in image coords."""
    B, _, H, W = out.shape
    obj = out[:, 0].sigmoid()
    cls = out[:, 1 : 1 + n_classes]
    reg = F.relu(out[:, 1 + n_classes :])  # ltrb in cell units
    results = []
    for b in range(B):
        boxes, scores, labels = [], [], []
        for y in range(H):
            for x in range(W):
                o = obj[b, y, x].item()
                if o < conf_thr:
                    continue
                logits = cls[b, :, y, x]
                c = int(torch.argmax(logits).item())
                # class score * obj
                sc = o * torch.softmax(logits, dim=0)[c].item()
                if sc < conf_thr:
                    continue
                l, t, r, bb = reg[b, :, y, x]
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                x1 = cx - l.item() * stride
                y1 = cy - t.item() * stride
                x2 = cx + r.item() * stride
                y2 = cy + bb.item() * stride
                boxes.append([x1, y1, x2, y2])
                scores.append(sc)
                labels.append(c)
        if boxes:
            results.append(
                {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "scores": torch.tensor(scores, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.int64),
                }
            )
        else:
            results.append(
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.int64),
                }
            )
    return results


def build_centernet_targets(
    targets: List[Dict[str, torch.Tensor]],
    feat_h: int,
    feat_w: int,
    stride: int,
    n_classes: int,
    img_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns obj_t [B,H,W], cls_t [B,H,W] long (-1 ignore/0..C-1), reg_t [B,4,H,W]
    Assign GT to cell containing center.
    """
    B = len(targets)
    obj_t = torch.zeros(B, feat_h, feat_w)
    cls_t = torch.full((B, feat_h, feat_w), -1, dtype=torch.long)
    reg_t = torch.zeros(B, 4, feat_h, feat_w)
    for b, t in enumerate(targets):
        boxes = t["boxes"]
        labels = t["labels"]
        for box, lab in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            gx = int(cx / stride)
            gy = int(cy / stride)
            if not (0 <= gx < feat_w and 0 <= gy < feat_h):
                continue
            obj_t[b, gy, gx] = 1.0
            cls_t[b, gy, gx] = int(lab)
            # ltrb in stride units
            cell_cx = (gx + 0.5) * stride
            cell_cy = (gy + 0.5) * stride
            reg_t[b, 0, gy, gx] = (cell_cx - x1) / stride
            reg_t[b, 1, gy, gx] = (cell_cy - y1) / stride
            reg_t[b, 2, gy, gx] = (x2 - cell_cx) / stride
            reg_t[b, 3, gy, gx] = (y2 - cell_cy) / stride
    return obj_t, cls_t, reg_t


def centernet_loss(out: torch.Tensor, targets: List[Dict], n_classes: int = 3, stride: int = 8):
    B, _, H, W = out.shape
    obj_t, cls_t, reg_t = build_centernet_targets(targets, H, W, stride, n_classes, 64)
    device = out.device
    obj_t, cls_t, reg_t = obj_t.to(device), cls_t.to(device), reg_t.to(device)
    obj_p = out[:, 0]
    cls_p = out[:, 1 : 1 + n_classes]
    reg_p = out[:, 1 + n_classes :]
    loss_obj = F.binary_cross_entropy_with_logits(obj_p, obj_t)
    pos = obj_t > 0.5
    if pos.any():
        loss_cls = F.cross_entropy(cls_p.permute(0, 2, 3, 1)[pos], cls_t[pos])
        loss_reg = F.l1_loss(F.relu(reg_p.permute(0, 2, 3, 1)[pos]), reg_t.permute(0, 2, 3, 1)[pos])
    else:
        loss_cls = out.sum() * 0
        loss_reg = out.sum() * 0
    loss = loss_obj + loss_cls + loss_reg
    return loss, {
        "loss_obj": float(loss_obj.detach()),
        "loss_cls": float(loss_cls.detach()),
        "loss_reg": float(loss_reg.detach()),
        "loss": float(loss.detach()),
    }


class TwoStageLite(nn.Module):
    """
    Pedagogical two-stage: backbone + RPN-like objectness on grid + RoI-free
    second head that refines using pooled global features per peak.
    Still center-based but exposes rpn_loss vs head_loss (P2 narrative).
    """

    def __init__(self, n_classes: int = 3, stride: int = 8):
        super().__init__()
        self.n_classes = n_classes
        self.stride = stride
        self.backbone = TinyBackbone(64)
        self.rpn = nn.Conv2d(64, 1, 1)  # objectness proposal
        self.head = nn.Conv2d(64, n_classes + 4, 1)

    def forward(self, x):
        f = self.backbone(x)
        rpn = self.rpn(f)
        head = self.head(f)
        return rpn, head, f


def two_stage_lite_loss(rpn, head, targets, n_classes=3, stride=8):
    B, _, H, W = rpn.shape
    obj_t, cls_t, reg_t = build_centernet_targets(targets, H, W, stride, n_classes, 64)
    device = rpn.device
    obj_t, cls_t, reg_t = obj_t.to(device), cls_t.to(device), reg_t.to(device)
    loss_rpn = F.binary_cross_entropy_with_logits(rpn[:, 0], obj_t)
    pos = obj_t > 0.5
    cls_p = head[:, :n_classes]
    reg_p = head[:, n_classes:]
    if pos.any():
        loss_cls = F.cross_entropy(cls_p.permute(0, 2, 3, 1)[pos], cls_t[pos])
        loss_reg = F.l1_loss(F.relu(reg_p.permute(0, 2, 3, 1)[pos]), reg_t.permute(0, 2, 3, 1)[pos])
    else:
        loss_cls = head.sum() * 0
        loss_reg = head.sum() * 0
    loss = loss_rpn + loss_cls + loss_reg
    return loss, {
        "loss_rpn": float(loss_rpn.detach()),
        "loss_cls": float(loss_cls.detach()),
        "loss_reg": float(loss_reg.detach()),
        "loss": float(loss.detach()),
    }


def decode_two_stage(rpn, head, stride=8, conf_thr=0.3, n_classes=3):
    obj = rpn[:, 0].sigmoid()
    # pack like centernet
    out = torch.cat([rpn, head], dim=1)
    return decode_centernet(out, stride=stride, conf_thr=conf_thr, n_classes=n_classes)


class DETRLite(nn.Module):
    """Tiny DETR: CNN backbone -> flatten -> transformer encoder/decoder -> box+class."""

    def __init__(self, n_classes=3, n_queries=8, d_model=64, nhead=4, img_size=64):
        super().__init__()
        self.n_classes = n_classes
        self.n_queries = n_queries
        self.backbone = TinyBackbone(d_model)
        self.input_proj = nn.Conv2d(d_model, d_model, 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=1)
        self.query_embed = nn.Embedding(n_queries, d_model)
        self.class_embed = nn.Linear(d_model, n_classes + 1)  # + no-object
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4), nn.Sigmoid()
        )
        self.img_size = img_size

    def forward(self, x):
        f = self.input_proj(self.backbone(x))  # B,C,H,W
        B, C, H, W = f.shape
        mem = f.flatten(2).permute(0, 2, 1)  # B,HW,C
        mem = self.encoder(mem)
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.decoder(q, mem)
        logits = self.class_embed(hs)  # B,Q,C+1
        boxes = self.bbox_embed(hs)  # B,Q,4 cxcywh normalized
        return logits, boxes


def detr_lite_loss(logits, boxes_norm, targets, img_size=64, n_classes=3):
    """Hungarian match then CE + L1 on matched pairs; no-object on unmatched queries."""
    from assignment import hungarian_match
    from boxes import cxcywh_to_xyxy, xyxy_to_cxcywh
    import numpy as np

    B, Q, _ = logits.shape
    device = logits.device
    total_loss = torch.tensor(0.0, device=device)
    logs = {"n_match": 0}
    for b in range(B):
        gt_boxes = targets[b]["boxes"].detach().cpu().numpy()
        gt_labels = targets[b]["labels"].detach().cpu().numpy()
        # pred boxes to xyxy absolute
        pb = boxes_norm[b].detach().cpu().numpy() * img_size
        pb_xyxy = cxcywh_to_xyxy(pb)
        plogits = logits[b].detach().cpu().numpy()
        if len(gt_boxes) == 0:
            # all no-object
            tgt = torch.full((Q,), n_classes, device=device, dtype=torch.long)
            total_loss = total_loss + F.cross_entropy(logits[b], tgt)
            continue
        match = hungarian_match(pb_xyxy, plogits, gt_boxes, gt_labels, n_classes + 1)
        pred_idx = match["pred_idx"]
        gt_idx = match["gt_idx"]
        logs["n_match"] += len(pred_idx)
        # class targets
        tgt_cls = torch.full((Q,), n_classes, device=device, dtype=torch.long)
        tgt_box = boxes_norm[b].detach().clone()
        for pi, gi in zip(pred_idx, gt_idx):
            tgt_cls[int(pi)] = int(gt_labels[int(gi)])
            g_cxcywh = xyxy_to_cxcywh(gt_boxes[int(gi) : int(gi) + 1])[0] / img_size
            tgt_box[int(pi)] = torch.tensor(g_cxcywh, device=device, dtype=boxes_norm.dtype)
        loss_cls = F.cross_entropy(logits[b], tgt_cls)
        if len(pred_idx):
            pi = torch.tensor(pred_idx, device=device, dtype=torch.long)
            loss_box = F.l1_loss(boxes_norm[b][pi], tgt_box[pi])
        else:
            loss_box = boxes_norm[b].sum() * 0
        total_loss = total_loss + loss_cls + 5.0 * loss_box
    return total_loss / B, logs


def decode_detr(logits, boxes_norm, img_size=64, conf_thr=0.3, n_classes=3):
    B, Q, _ = logits.shape
    prob = torch.softmax(logits, dim=-1)
    results = []
    for b in range(B):
        scores, labels = prob[b, :, :n_classes].max(dim=-1)
        keep = scores > conf_thr
        # also prefer not no-object: already took max over real classes
        boxes = boxes_norm[b][keep]
        from boxes import cxcywh_to_xyxy
        import numpy as np

        if len(boxes) == 0:
            results.append(
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.int64),
                }
            )
            continue
        xyxy = torch.tensor(
            cxcywh_to_xyxy((boxes.detach().cpu().numpy() * img_size)),
            dtype=torch.float32,
        )
        results.append(
            {
                "boxes": xyxy,
                "scores": scores[keep].detach().cpu(),
                "labels": labels[keep].detach().cpu(),
            }
        )
    return results
