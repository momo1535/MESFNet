# losses4.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 基础工具函数
# =========================
def dice_loss_logits(logits, target, eps=1.0):
    """
    Dice loss with logits input
    """
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    union = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (union + eps)


def soft_iou_loss_logits(logits, target, eps=1.0):
    """Batch-global soft IoU loss, aligned with the competition metric."""
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum()
    union = prob.sum() + target.sum() - intersection
    return 1.0 - (intersection + eps) / (union + eps)


def soft_edge_map(prob):
    """
    可微分的软边缘提取（Sobel-like）
    prob: (B,1,H,W) or (B,H,W)
    """
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)

    # Sobel kernels
    kernel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        device=prob.device, dtype=prob.dtype
    ).view(1, 1, 3, 3)

    kernel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        device=prob.device, dtype=prob.dtype
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(prob, kernel_x, padding=1)
    grad_y = F.conv2d(prob, kernel_y, padding=1)

    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return edge


def morphology_aux_loss(mask_logits, target_mask):
    """
    简化版形态辅助约束：
    约束预测 mask 与 GT 在整体连通性上的一致性
    """
    prob = torch.sigmoid(mask_logits)
    return F.l1_loss(prob, target_mask)


# =========================
# 总损失函数
# =========================
class TotalLoss(nn.Module):
    def __init__(
        self,
        # w_mask=1.0,
        # w_edge=0.8,
        # w_morph=0.01,
        # w_aux=0.01,
        # lambda_cons=0.02
        w_mask = 0.8,
        w_iou = 0.0,
        w_edge = 0.15,
        w_morph = 0.025,
        w_aux = 0.025,
        lambda_cons = 0.1,
        morph_loss_type = "mse",
    ):
        super().__init__()

        self.w_mask = w_mask
        self.w_iou = w_iou
        self.w_edge = w_edge
        self.w_morph = w_morph
        self.w_aux = w_aux
        self.lambda_cons = lambda_cons
        self.morph_loss_type = morph_loss_type

        if self.morph_loss_type not in {"mse", "smooth_l1"}:
            raise ValueError(f"Unsupported morph loss: {self.morph_loss_type}")

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, target_mask, target_edge, target_morph):
        """
        outputs: dict with keys:
            - "mask": (B,1,H,W)
            - "edge": (B,1,H,W)
            - "morph": (B,K)
        """

        # -------- unpack --------
        mask_logits = outputs["mask"]
        edge_logits = outputs["edge"]
        morph_pred = outputs["morph"]

        # 保证 shape
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)
        if target_edge.dim() == 3:
            target_edge = target_edge.unsqueeze(1)

        target_mask = target_mask.float()
        target_edge = target_edge.float()
        target_morph = target_morph.float()

        # ===== 1. 主分割 loss =====
        l_mask = (
            self.bce(mask_logits, target_mask)
            + dice_loss_logits(mask_logits, target_mask)
        )

        # ===== 2. 边缘 loss（BCE + Dice）=====
        l_edge = (
            self.bce(edge_logits, target_edge)
            + dice_loss_logits(edge_logits, target_edge)
        )

        # ===== 3. 形态回归 loss =====
        if self.morph_loss_type == "smooth_l1":
            l_morph = F.smooth_l1_loss(morph_pred, target_morph)
        else:
            l_morph = F.mse_loss(morph_pred, target_morph)

        # ===== 4. 形态辅助约束（弱）=====
        l_aux = morphology_aux_loss(mask_logits, target_mask)

        # ===== 5. 基础 loss 汇总 =====
        total_loss = (
            self.w_mask * l_mask
            + self.w_iou * soft_iou_loss_logits(mask_logits, target_mask)
            + self.w_edge * l_edge
            + self.w_morph * l_morph
            + self.w_aux * l_aux
        )

        # ===== 6. ★ 边界一致性约束（减少欠分割的关键）=====
        if self.lambda_cons > 0:
            mask_prob = torch.sigmoid(mask_logits)
            mask_soft_edge = soft_edge_map(mask_prob)
            l_cons = F.l1_loss(mask_soft_edge, target_edge)
            total_loss = total_loss + self.lambda_cons * l_cons

        return total_loss

    def set_morph_weight(self, weight):
        self.w_morph = float(weight)
