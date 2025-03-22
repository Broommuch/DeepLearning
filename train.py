import argparse
from matplotlib import testing
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os, time, random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from dataloader import TinySegData
from evaluate import plot_metrics, plot_confusion_matrix

from metrics import get_confusion_matrix_for_3d, compute_class_ious

from utils import CombinedLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="语义分割训练脚本")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--val_batch_size", type=int, default=32, help="验证批大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument(
        "--model",
        type=str,
        default="pspnet",
        choices=["pspnet", "deeplabv3", "ccnet"],
        help="分割模型",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--early_stop", type=int, default=10, help="提前停止轮数")
    parser.add_argument(
        "--aug_type",
        type=str,
        default="advanced",
        choices=["none", "base", "advanced"],
        help="数据增强类型",
    )
    return parser.parse_args()


def create_directories(model_type, aug_type):
    """创建日志和保存目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/{model_type}_{aug_type}_{timestamp}"
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    vis_dir = os.path.join(log_dir, "visualizations")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "training.log")

    return log_dir, ckpt_dir, vis_dir, log_file


def get_model(model_name, num_classes=6, pretrained=True):
    """获取指定的模型"""
    if model_name == "pspnet":
        from models.pspnet import PSPNet

        return PSPNet(n_classes=num_classes, pretrained=pretrained)
    elif model_name == "deeplabv3":
        from models.deeplabv3 import build_deeplabv3_custom

        return build_deeplabv3_custom(num_classes=num_classes)

    elif model_name == "ccnet":
        from models.ccnet import build_ccnet

        return build_ccnet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch,
    total_epochs,
):
    """训练一个epoch"""
    model.train()
    epoch_iou = []
    epoch_loss = []
    epoch_start = time.time()

    prog_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for j, (images, seg_gts, rets) in enumerate(prog_bar):
        images = images.to(device)
        seg_gts = seg_gts.to(device)
        optimizer.zero_grad()

        seg_logit = model(images)
        loss_seg = criterion(seg_logit, seg_gts.long())
        loss = loss_seg
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        prog_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        if j % 10 == 0:
            seg_preds = torch.argmax(seg_logit, dim=1)
            seg_preds_np = seg_preds.detach().cpu().numpy()
            seg_gts_np = seg_gts.cpu().numpy()

            confusion_matrix = get_confusion_matrix_for_3d(
                seg_gts_np, seg_preds_np, class_num=6
            )
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            IU_array = tp / np.maximum(1.0, pos + res - tp)
            mean_IU = IU_array.mean()

            log_str = "[E{}/{} - {}] ".format(epoch, epoch, j)
            log_str += "loss[seg]: {:0.4f}, miou: {:0.4f}, ".format(
                loss_seg.item(), mean_IU
            )
            print(log_str)

            images_np = np.transpose((images.cpu().numpy() + 1) * 127.5, (0, 2, 3, 1))
            n, h, w, c = images_np.shape
            images_np = images_np.reshape(n * h, w, -1)[:, :, 0]
            seg_preds_np = seg_preds_np.reshape(n * h, w)
            visual_np = np.concatenate([images_np, seg_preds_np * 40], axis=1)  # NH * W
            cv2.imwrite("visual.png", visual_np)
            epoch_iou.append(mean_IU)

    # 计算epoch统计信息
    epoch_iou_avg = np.mean(epoch_iou) if epoch_iou else 0
    epoch_loss_avg = np.mean(epoch_loss)
    epoch_time = round(time.time() - epoch_start, 2)

    return epoch_loss_avg, epoch_iou_avg, epoch_time


def validate(model, val_loader, criterion, device, class_num=6):
    """在验证集上评估模型"""
    model.eval()
    val_ious = []
    val_losses = []

    with torch.no_grad():
        for images, seg_gts, _ in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            seg_gts = seg_gts.to(device)

            seg_logit = model(images)
            loss_val = criterion(seg_logit, seg_gts.long()).item()
            val_losses.append(loss_val)

            seg_preds = torch.argmax(seg_logit, dim=1)
            seg_preds_np = seg_preds.cpu().numpy()
            seg_gts_np = seg_gts.cpu().numpy()

            confusion_matrix = get_confusion_matrix_for_3d(
                seg_gts_np, seg_preds_np, class_num=class_num
            )
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            IU_array = tp / np.maximum(1.0, pos + res - tp)
            val_ious.append(IU_array.mean())

    val_iou_avg = np.mean(val_ious)
    val_loss_avg = np.mean(val_losses)

    return val_loss_avg, val_iou_avg


def main():
    mp.set_start_method('spawn', force=True)  # 添加在 main() 开头
    args = get_args()
    set_seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> 使用设备: {device}")

    # 创建目录
    log_dir, ckpt_dir, vis_dir, log_file = create_directories(args.model, args.aug_type)

    # 定义日志函数
    def log_message(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log_message(f"=> 开始训练 模型: {args.model}, 批大小: {args.batch_size}")

    # 创建数据加载器
    # Create the full training dataset
    train_dataset = TinySegData(phase="train", aug_type=args.aug_type)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        TinySegData(phase="val", aug_type="none"),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 创建模型
    model = get_model(args.model)
    model = model.to(device)

    # 创建优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CombinedLoss(ce_weight=1, dice_weight=0)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    # 初始化指标记录和早停计数器
    best_iou = 0.0
    early_stop_counter = 0
    train_losses, train_mious = [], []
    val_losses, val_mious = [], []

    # 训练循环
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss, train_iou, epoch_time = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            args.epochs,
        )

        train_losses.append(train_loss)
        train_mious.append(train_iou)

        log_message(
            f"=> Epoch {epoch} 完成，用时 {epoch_time}s，损失: {train_loss:.4f}，mIoU: {train_iou:.4f}"
        )
        # 验证
        val_loss, val_iou = validate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_mious.append(val_iou)

        log_message(f"=> 验证 - 损失: {val_loss:.4f}，mIoU: {val_iou:.4f}")

        # 学习率调整
        scheduler.step()
        log_message(f"=> 当前学习率: {scheduler.get_last_lr()[0]:.6f}")
        # 保存指标数据
        metrics_path = os.path.join(log_dir, "training_metrics.npz")
        np.savez(
            metrics_path,
            train_losses=train_losses,
            train_mious=train_mious,
            val_losses=val_losses,
            val_mious=val_mious,
        )

        # 绘制指标曲线
        plot_metrics(
            train_losses,
            train_mious,
            val_losses,
            val_mious,
            os.path.join(log_dir, "metrics.png"),
        )

        # 保存检查点
        if epoch % 10 == 0 or epoch == args.epochs - 1 or val_iou > best_iou:
            checkpoint_path = os.path.join(
                ckpt_dir, f"epoch_{epoch}_iou{val_iou:.2f}.pth"
            )
            log_message(f"=> 保存检查点到 {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

            if val_iou > best_iou:
                best_iou = val_iou
                best_model_path = os.path.join(ckpt_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                log_message(f"=> 新的最佳模型，mIoU: {best_iou:.4f}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1

        # 早停机制
        if early_stop_counter >= args.early_stop:
            log_message(f"=> 验证集性能 {args.early_stop} 个epoch未提升，提前停止训练")
            break

    log_message("=> 训练完成，在验证集上进行最终评估...")
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    final_val_loss, final_val_iou = validate(model, test_loader, criterion, device)
    log_message(
        f"=> 最终评估结果 - mIoU: {final_val_iou:.4f}, 损失: {final_val_loss:.4f}"
    )

    # 计算最终的混淆矩阵
    confusion_matrix_total = np.zeros((6, 6), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for images, seg_gts, _ in tqdm(test_loader, desc="计算混淆矩阵"):
            images = images.to(device)
            seg_gts = seg_gts.to(device)

            seg_logit = model(images)
            seg_preds = torch.argmax(seg_logit, dim=1)

            seg_preds_np = seg_preds.cpu().numpy()
            seg_gts_np = seg_gts.cpu().numpy()

            batch_cm = get_confusion_matrix_for_3d(
                seg_gts_np, seg_preds_np, class_num=6
            )
            confusion_matrix_total += batch_cm

    # 绘制并保存混淆矩阵
    class_names = ["background", "person", "bird", "car", "cat", "plane"]
    cm_save_path = os.path.join(vis_dir, "confusion_matrix.png")
    confusion_matrix_total = confusion_matrix_total.astype(np.int64)
    plot_confusion_matrix(confusion_matrix_total, class_names, cm_save_path)
    log_message(f"=> 混淆矩阵已保存到 {cm_save_path}")

    # 计算类别IoU
    class_ious = compute_class_ious(model, test_loader, device)
    for i, class_iou in enumerate(class_ious):
        log_message(f"   类别 {i} IoU: {class_iou:.4f}")

    log_message(f"=> 训练完成！最佳模型保存在 {best_model_path}")


if __name__ == "__main__":
    main()
