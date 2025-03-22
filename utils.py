import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        计算Dice Loss
        Args:
            logits: 模型输出的 logits，形状为 [batch_size, num_classes, height, width]
            targets: 目标标签，形状为 [batch_size, height, width]
        """
        num_classes = logits.shape[1]

        # 将 targets 转换成 one-hot 编码
        # 首先，将 targets 展平
        targets_flat = targets.view(-1)

        # 创建 one-hot 编码 targets
        targets_one_hot = torch.zeros(targets_flat.size(0), num_classes).to(logits.device)
        targets_one_hot.scatter_(1, targets_flat.unsqueeze(1), 1)

        # 重新塑形为 [batch_size, height, width, num_classes]
        targets_one_hot = targets_one_hot.view(*targets.shape, -1)
        # 将通道维度移到正确位置 [batch_size, num_classes, height, width]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

        # 计算softmax概率
        probs = F.softmax(logits, dim=1)

        # 展平 probs 和 one-hot targets
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        targets_one_hot_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)

        # 计算每个类别的 Dice 系数
        intersection = (probs_flat * targets_one_hot_flat).sum(2)
        denominator = probs_flat.sum(2) + targets_one_hot_flat.sum(2)

        # 聚合所有类别，避免未出现类别影响损失
        dice_score = (2.0 * intersection.sum(1) + self.smooth) / (denominator.sum(1) + self.smooth)
        dice_loss = 1.0 - dice_score

        # 计算批次平均 Dice Loss
        return dice_loss.mean()


class CombinedLoss(nn.Module):
    """
    结合交叉熵和Dice Loss的损失函数
    """

    def __init__(self, ce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
