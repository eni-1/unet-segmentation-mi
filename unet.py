import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.model_selection import train_test_split
from torchvision import transforms
from scipy.ndimage import zoom


class Normalize:
    def __call__(self, image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))


class Tensor:
    def __call__(self, array):
        return torch.tensor(array, dtype=torch.float32).unsqueeze(0)


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, array):
        factors = [ns / os for ns, os in zip(self.size, array.shape)]
        return zoom(array, factors, order=1)


class Checkpoint:
    def __init__(self, filepath='best_model.pth', monitor='val_loss', mode='min'):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def save(self, current_score):
        if self.mode == 'min':
            if current_score < self.best_score:
                self.best_score = current_score
                return True
        else:
            if current_score > self.best_score:
                self.best_score = current_score
                return True
        return False


class Loss:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.dice_scores = []

    def update(self, train_loss, val_loss, dice_score):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.dice_scores.append(dice_score)

    def plt_loss(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.plot(self.train_losses, label='Train Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.subplot(132)
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.subplot(133)
        plt.plot(self.dice_scores, label='Dice Score', color='green')
        plt.title('Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()


class Metrics:
    def __init__(self):
        self.l1_dcs = []
        self.l1_hsd = []

    def update(self, prediction, ground_truth):
        pred_binary = (prediction > 0.5).float()
        gt_binary = ground_truth.float()
        intersection = torch.sum(pred_binary * gt_binary)
        union = torch.sum(pred_binary) + torch.sum(gt_binary)
        dice = (2. * intersection) / (union + 1e-8)
        self.l1_dcs.append(dice.item())
        surface_pred = self.surface_points(pred_binary)
        surface_gt = self.surface_points(gt_binary)
        max_distance = self.distance(surface_pred, surface_gt)
        hsd = 1.0 - (max_distance / (max_distance + 1e-8))
        self.l1_hsd.append(hsd)

    @staticmethod
    def surface_points(binary_mask):
        kernel = torch.ones(1, 1, 3, 3, 3).to(binary_mask.device)
        boundary = F.conv3d(binary_mask.unsqueeze(0), kernel, padding=1)
        surface_points = (boundary > 0) & (boundary < 27)
        return surface_points.float()

    @staticmethod
    def distance(surface1, surface2):
        coords1 = torch.nonzero(surface1)
        coords2 = torch.nonzero(surface2)
        if len(coords1) == 0 or len(coords2) == 0:
            return float('inf')
        distances = torch.cdist(coords1.float(), coords2.float())
        max_min_dist = torch.max(torch.min(distances, dim=1)[0])
        return max_min_dist.item()

    def test_metrics(self):
        metrics = {
            "DCS": {str(i): score for i, score in enumerate(self.l1_dcs)},
            "sHSD": {str(i): score for i, score in enumerate(self.l1_hsd)},
            "aggregates": {
                "DCS": {
                    "max": max(self.l1_dcs),
                    "min": min(self.l1_dcs),
                    "std": np.std(self.l1_dcs),
                    "25pc": np.percentile(self.l1_dcs, 25),
                    "50pc": np.percentile(self.l1_dcs, 50),
                    "75pc": np.percentile(self.l1_dcs, 75),
                    "mean": np.mean(self.l1_dcs),
                    "count": len(self.l1_dcs)
                },
                "HSD": {
                    "max": max(self.l1_hsd),
                    "min": min(self.l1_hsd),
                    "std": np.std(self.l1_hsd),
                    "25pc": np.percentile(self.l1_hsd, 25),
                    "50pc": np.percentile(self.l1_hsd, 50),
                    "75pc": np.percentile(self.l1_hsd, 75),
                    "mean": np.mean(self.l1_hsd),
                    "count": len(self.l1_hsd)
                }
            }
        }
        return metrics


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dropout_rate=0.0, instance_norm=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        if instance_norm:
            self.norm = nn.InstanceNorm3d(out_channels, eps=1e-5, momentum=0.1, affine=True)
        else:
            self.norm = nn.BatchNorm3d(out_channels)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        x = self.dropout(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock(out_channels, out_channels,
                               dropout_rate=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, n_classes=1):
        super().__init__()
        self.encoder1 = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ResBlock(base_channels, base_channels)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 2)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4)
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bridge = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8),
            ResBlock(base_channels * 8, base_channels * 8)
        )
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4,
                                          kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 4),
            ResBlock(base_channels * 4, base_channels * 4)
        )
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2,
                                          kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 2),
            ResBlock(base_channels * 2, base_channels * 2)
        )
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels,
                                          kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels),
            ResBlock(base_channels, base_channels)
        )
        self.final_conv = nn.Conv3d(base_channels, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bridge = self.bridge(self.pool3(enc3))
        dec3 = self.upconv3(bridge)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        output = self.final_conv(dec1)
        return self.sigmoid(output)


class Images(Dataset):
    def __init__(self, image_paths, label_paths=None, target_size=(128, 128, 128), train=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.train = train
        self.image_transforms = transforms.Compose([
            Resize(target_size),
            Normalize(),
            Tensor()
        ])
        if self.train:
            self.label_transforms = transforms.Compose([
                Resize(target_size),
                Tensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata()
        image = self.image_transforms(image)

        if self.train:
            label = nib.load(self.label_paths[idx]).get_fdata()
            label = self.label_transforms(label)
            return image, label
        else:
            return image, idx


def dice_loss(prediction, target):
    smooth = 1.0
    prediction = prediction.view(-1)
    target = target.view(-1)
    intersection = (prediction * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (prediction.sum() + target.sum() + smooth))


def bce_loss(prediction, target, weights=(0.5, 0.5)):
    dice = dice_loss(prediction, target)
    bce = F.binary_cross_entropy(prediction, target)
    return weights[0] * dice + weights[1] * bce


def train(train_images, train_labels, val_images, val_labels, test_images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    loss_tracker = Loss()
    model_checkpoint = Checkpoint()

    train_dataset = Images(train_images, train_labels, train=True)
    val_dataset = Images(val_images, val_labels, train=True)
    test_dataset = Images(test_images, label_paths=None, train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = bce_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        val_loss = 0
        metrics_tracker = Metrics()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += bce_loss(outputs, masks).item()
                for pred, gt in zip(outputs, masks):
                    metrics_tracker.update(pred, gt)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_dice_score = np.mean(metrics_tracker.l1_dcs)
        loss_tracker.update(avg_train_loss, avg_val_loss, avg_dice_score)
        scheduler.step(avg_val_loss)
        if model_checkpoint.save(avg_val_loss):
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': avg_val_loss
            }, model_checkpoint.filepath)
        print(f'epoch [{epoch + 1}/{epochs}], '
              f'train loss: {avg_train_loss:.4f}, '
              f'val loss: {avg_val_loss:.4f}, '
              f'dice score: {avg_dice_score:.4f}')
    loss_tracker.plt_loss()
    return model


def main():
    base_dir = "path"
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        dataset_info = json.load(f)
    # .json: "training":[{"image":"./images/19.nii.gz","label":"./labels/19.nii.gz"}]
    train_images = [os.path.join(base_dir, case['image'].replace('./', '')) for case in dataset_info['training']]
    train_labels = [os.path.join(base_dir, case['label'].replace('./', '')) for case in dataset_info['training']]
    test_images = [os.path.join(base_dir, img_path.replace('./', '')) for img_path in dataset_info['test']]

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels,
        test_size=6 / 41,
        random_state=1
    )

    print(f"training samples: {len(train_images)}")
    print(f"validation samples: {len(val_images)}")
    print(f"test samples: {len(test_images)}")
    os.makedirs('checkpoints', exist_ok=True)
    model = train(train_images, train_labels, val_images, val_labels, test_images)
    print("complete")
    return model


if __name__ == "__main__":
    main()
