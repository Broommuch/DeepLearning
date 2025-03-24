import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP冲突
os.environ["OMP_NUM_THREADS"] = "1"  # 禁用OpenMP多线程
import sys
import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

# 强制设置多进程启动方式
mp.set_start_method('spawn', force=True)

# 自定义模块
from dataloader import TinySegData
from utils import CombinedLoss
from metrics import get_confusion_matrix_for_3d


class SafeDataLoader:
    """安全的数据加载器封装"""

    def __init__(self, args):
        self.args = args
        self._init_datasets()

    def _init_datasets(self):
        """延迟初始化数据集以避免序列化问题"""
        self.train_dataset = TinySegData(phase="train", aug_type=self.args.aug_type)
        self.val_dataset = TinySegData(phase="val", aug_type="none")

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=min(4, self.args.num_workers),  # 限制最大workers
            worker_init_fn=self.worker_init,
            persistent_workers=True
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.val_batch_size,
            num_workers=0  # 验证集禁用多进程
        )

    @staticmethod
    def worker_init(worker_id):
        """安全的工作进程初始化"""
        seed = int(torch.initial_seed()) % 2 ** 32
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)


class TrainingSystem:
    """训练系统封装"""

    def __init__(self, args):
        self.args = args
        self._init_directories()
        self._init_model()
        self._init_optimizer()
        self.data_loader = SafeDataLoader(args)

    def _init_directories(self):
        """初始化输出目录"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"runs/{self.args.model}_{self.args.aug_type}_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = open(os.path.join(self.log_dir, "training.log"), "a", buffering=1)

    def _init_model(self):
        """初始化模型"""
        model_mapping = {
            "pspnet": self._get_pspnet,
            "deeplabv3": self._get_deeplabv3,
            "ccnet": self._get_ccnet
        }
        self.model = model_mapping[self.args.model]().to(self.device)

    def _get_pspnet(self):
        from models.pspnet import PSPNet
        return PSPNet(n_classes=6)

    def _get_deeplabv3(self):
        from models.deeplabv3 import build_deeplabv3_custom
        return build_deeplabv3_custom(num_classes=6)

    def _get_ccnet(self):
        from models.ccnet import build_ccnet
        return build_ccnet(num_classes=6)

    def _init_optimizer(self):
        """初始化优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=1e-4
        )
        self.criterion = CombinedLoss(ce_weight=1, dice_weight=0)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs
        )

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(self, message):
        """线程安全的日志记录"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        self.log_file.write(msg + "\n")

    def safe_visualization(self, images, preds, epoch, batch_idx):
        """安全的可视化保存"""
        try:
            img = ((images[0].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            pred = preds[0].cpu().numpy().astype(np.uint8) * 40
            cv2.imwrite(
                os.path.join(self.log_dir, f"epoch{epoch}_batch{batch_idx}.png"),
                np.hstack([img, pred])
            )
        except Exception as e:
            self.log(f"可视化保存失败: {str(e)}")

    def train_epoch(self, epoch):
        """单epoch训练"""
        self.model.train()
        loader = self.data_loader.get_train_loader()
        total_loss, total_iou = 0, 0

        for batch_idx, (images, seg_gts, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            images, seg_gts = images.to(self.device), seg_gts.to(self.device)

            # 梯度更新
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, seg_gts.long())
            loss.backward()
            self.optimizer.step()

            # 指标计算
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                confusion = get_confusion_matrix_for_3d(
                    seg_gts.cpu().numpy(),
                    preds.cpu().numpy(),
                    6
                )
                tp = np.diag(confusion)
                miou = (tp / np.maximum(1.0, confusion.sum(1) + confusion.sum(0) - tp)).mean()

            total_loss += loss.item()
            total_iou += miou

            # 安全保存
            if batch_idx % 10 == 0:
                self.safe_visualization(images, preds, epoch, batch_idx)

        return total_loss / len(loader), total_iou / len(loader)

    def validate(self):
        """验证过程"""
        self.model.eval()
        loader = self.data_loader.get_val_loader()
        total_loss, total_iou = 0, 0

        with torch.no_grad():
            for images, seg_gts, _ in loader:
                images, seg_gts = images.to(self.device), seg_gts.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, seg_gts.long())

                preds = torch.argmax(outputs, dim=1)
                confusion = get_confusion_matrix_for_3d(
                    seg_gts.cpu().numpy(),
                    preds.cpu().numpy(),
                    6
                )
                tp = np.diag(confusion)
                miou = (tp / np.maximum(1.0, confusion.sum(1) + confusion.sum(0) - tp)).mean()

                total_loss += loss.item()
                total_iou += miou

        return total_loss / len(loader), total_iou / len(loader)

    def run(self):
        """主训练循环"""
        best_iou = 0.0
        early_stop = 0

        for epoch in range(self.args.epochs):
            start_time = time.time()

            # 训练阶段
            train_loss, train_iou = self.train_epoch(epoch)

            # 验证阶段
            val_loss, val_iou = self.validate()

            # 学习率调整
            self.scheduler.step()

            # 记录日志
            self.log(f"Epoch {epoch + 1}/{self.args.epochs} | "
                     f"Train Loss: {train_loss:.4f} mIoU: {train_iou:.4f} | "
                     f"Val Loss: {val_loss:.4f} mIoU: {val_iou:.4f} | "
                     f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # 早停机制
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), os.path.join(self.log_dir, "best_model.pth"))
                early_stop = 0
            else:
                early_stop += 1

            if early_stop >= self.args.early_stop:
                self.log(f"Early stopping at epoch {epoch + 1}")
                break


def parse_args():
    parser = argparse.ArgumentParser(description="安全训练配置")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model", choices=["pspnet", "deeplabv3", "ccnet"], default="pspnet")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--aug_type", choices=["none", "base", "advanced"], default="advanced")
    return parser.parse_args()


if __name__ == "__main__":
    # 确保在Windows下正确初始化
    if sys.platform.startswith('win'):
        mp.freeze_support()

    args = parse_args()
    system = TrainingSystem(args)
    system.run()