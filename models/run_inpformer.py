"""
INP-Former wrapper script for thesis benchmark.

This script trains INP-Former on Real-IAD and saves raw anomaly scores
to a .pkl file that can be loaded by the evaluation pipeline.

Usage in Colab:
    python models/run_inpformer.py \
        --data_root /content/realiad \
        --output_path /content/drive/MyDrive/results/inpformer_scores.pkl \
        --phase train

    python models/run_inpformer.py \
        --data_root /content/realiad \
        --output_path /content/drive/MyDrive/results/inpformer_scores.pkl \
        --phase test
"""

import os
import sys
import pickle
import argparse
import torch
import numpy as np

# Add INP-Former to path
INP_FORMER_DIR = os.path.join(os.path.dirname(__file__), 'inp_former')
sys.path.insert(0, INP_FORMER_DIR)

# INP-Former imports
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from dataset import RealIADDataset, get_data_transforms
from utils import (evaluation_batch, WarmCosineScheduler,
                   global_cosine_hm_adaptive, setup_seed, get_logger)
from optimizers import StableAdamW


def build_model(encoder_name: str, inp_num: int, device: str):
    """Build INP-Former model with specified encoder."""
    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(inp_num, embed_dim))])
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])

    model = INP_Former(
        encoder=encoder,
        bottleneck=Bottleneck,
        aggregation=INP_Extractor,
        decoder=INP_Guided_Decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=INP
    ).to(device)

    return model, nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])


def get_raw_scores(model, dataloader, device):
    """
    Run inference and return raw per-image scores and per-pixel anomaly maps.
    This is a modified version of evaluation_batch that returns raw arrays
    instead of computing metrics — needed for our custom evaluation pipeline.
    """
    from torch.nn import functional as F
    from utils import get_gaussian_kernel, cal_anomaly_maps

    model.eval()
    image_paths = []
    gt_labels = []
    pred_scores = []
    anomaly_maps = []
    gt_masks = []

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(dataloader, ncols=80):
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]

            # Compute anomaly map
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = F.interpolate(
                anomaly_map, size=256, mode='bilinear', align_corners=False)
            gt_resized = F.interpolate(gt, size=256, mode='nearest')
            anomaly_map = gaussian_kernel(anomaly_map)

            # Image-level score: mean of top 1% pixels
            flat = anomaly_map.flatten(1)
            k = max(1, int(flat.shape[1] * 0.01))
            score = torch.topk(flat, k, dim=1)[0].mean(dim=1)

            for i in range(len(img_path)):
                image_paths.append(img_path[i])
                gt_labels.append(int(label[i].item()))
                pred_scores.append(float(score[i].item()))
                anomaly_maps.append(
                    anomaly_map[i, 0].cpu().numpy().astype(np.float32))
                gt_masks.append(
                    gt_resized[i, 0].cpu().numpy().astype(np.uint8))

    return {
        'image_paths': image_paths,
        'gt_labels': gt_labels,
        'pred_scores': pred_scores,
        'anomaly_maps': anomaly_maps,
        'gt_masks': gt_masks
    }


def main(args):
    setup_seed(1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    data_transform, gt_transform = get_data_transforms(
        args.input_size, args.crop_size)

    # Build datasets for all Real-IAD categories
    train_data_list = []
    test_data_list = []

    for item in args.categories:
        train_data = RealIADDataset(
            root=args.data_root, category=item,
            transform=data_transform, gt_transform=gt_transform,
            phase='train'
        )
        test_data = RealIADDataset(
            root=args.data_root, category=item,
            transform=data_transform, gt_transform=gt_transform,
            phase='test'
        )
        train_data_list.append(train_data)
        test_data_list.append((item, test_data))

    # Build model
    model, trainable = build_model(args.encoder, args.inp_num, device)

    if args.phase == 'train':
        train_data = ConcatDataset(train_data_list)
        train_dataloader = DataLoader(
            train_data, batch_size=args.batch_size,
            shuffle=True, num_workers=4, drop_last=True
        )

        # Initialise weights
        from torch.nn.init import trunc_normal_
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW(
            [{'params': trainable.parameters()}],
            lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4,
            amsgrad=True, eps=1e-10
        )
        lr_scheduler = WarmCosineScheduler(
            optimizer, base_value=1e-3, final_value=1e-4,
            total_iters=args.total_epochs * len(train_dataloader),
            warmup_iters=100
        )

        print(f"Training on {len(train_data)} images for {args.total_epochs} epochs")
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                en, de, g_loss = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss + 0.2 * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()
            print(f"Epoch [{epoch+1}/{args.total_epochs}], "
                  f"loss: {np.mean(loss_list):.4f}")

        # Save model weights
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'inpformer_realiad.pth'))
        print(f"Model saved to {args.save_dir}/inpformer_realiad.pth")

    elif args.phase == 'test':
        # Load trained weights
        weights_path = os.path.join(args.save_dir, 'inpformer_realiad.pth')
        model.load_state_dict(torch.load(weights_path), strict=True)
        print(f"Loaded weights from {weights_path}")

        # Run inference and collect raw scores per category
        all_results = {}
        for category, test_data in test_data_list:
            print(f"\nEvaluating {category}...")
            test_dataloader = DataLoader(
                test_data, batch_size=args.batch_size,
                shuffle=False, num_workers=4
            )
            results = get_raw_scores(model, test_dataloader, device)
            all_results[category] = results
            print(f"  {category}: {len(results['image_paths'])} images, "
                  f"mean score: {np.mean(results['pred_scores']):.4f}")

        # Save raw scores to pkl file
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nRaw scores saved to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='INP-Former wrapper for thesis benchmark')

    # Data paths
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Real-IAD root directory')
    parser.add_argument('--output_path', type=str,
                        default='results/inpformer_scores.pkl',
                        help='Path to save raw scores')
    parser.add_argument('--save_dir', type=str,
                        default='results/inpformer_weights',
                        help='Directory to save/load model weights')

    # Model config — matches published paper defaults
    parser.add_argument('--encoder', type=str,
                        default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--inp_num', type=int, default=6)

    # Training config
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'])

    # Categories — all 30 Real-IAD categories by default
    parser.add_argument('--categories', nargs='+', default=[
        'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser',
        'fire_hood', 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut',
        'plastic_plug', 'porcelain_doll', 'regulator', 'rolled_strip_base',
        'sim_card_set', 'switch', 'tape', 'terminalblock', 'toothbrush',
        'toy', 'toy_brick', 'transistor1', 'usb', 'usb_adaptor', 'u_block',
        'vcpill', 'wooden_beads', 'woodstick', 'zipper'
    ])

    args = parser.parse_args()
    main(args)