from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from orin_occ.config import ProjectConfig
from orin_occ.data import build_dataset
from orin_occ.models import OrinOccNet


def build_model(config: ProjectConfig) -> nn.Module:
    return OrinOccNet(config)


def compute_occ_loss(
    logits: torch.Tensor,
    occupancy: torch.Tensor,
    mask_camera: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask_camera is None:
        return F.cross_entropy(logits, occupancy)

    class_count = logits.shape[1]
    flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, class_count)
    flat_target = occupancy.reshape(-1)
    flat_mask = mask_camera.reshape(-1).bool()

    if int(flat_mask.sum().item()) == 0:
        return F.cross_entropy(logits, occupancy)

    return F.cross_entropy(flat_logits[flat_mask], flat_target[flat_mask])


def train(config: ProjectConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = GradScaler(enabled=config.training.amp and torch.cuda.is_available())

    save_dir = Path(config.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(config.training.epochs):
        running_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            images = batch["images"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            extrinsics = batch["extrinsics"].to(device)
            lidar_bev = batch["lidar_bev"].to(device) if config.model.use_lidar else None
            occupancy = batch["occupancy"].to(device)
            mask_camera = batch["mask_camera"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(images, intrinsics, extrinsics, lidar_bev)
                loss = compute_occ_loss(logits, occupancy, mask_camera)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu())

            if step % 10 == 0 or step == len(loader):
                avg_loss = running_loss / step
                print(f"epoch={epoch + 1} step={step}/{len(loader)} loss={avg_loss:.4f}")

        checkpoint_path = save_dir / f"epoch_{epoch + 1}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config_name": config.project_name,
                "epoch": epoch + 1,
            },
            checkpoint_path,
        )

    print(f"training finished, checkpoints saved to {save_dir}")
