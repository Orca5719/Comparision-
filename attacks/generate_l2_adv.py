"""
Generate L2 adversarial examples on ImageNette using PGD.
Saves benign and adversarial images as uint8 pickle files.
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
import numpy as np
from tqdm import tqdm
import pickle
import argparse

DATA_DIR = 'PatchGuard/data/imagenette'
OUTPUT_DIR = 'shared_data'


def pgd_l2(model, x, y, epsilon=1.0, alpha=0.1, steps=40):
    """PGD with L2 norm constraint. x in [0,1], normalized internally."""
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            grad = x_adv.grad
            grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1)
            grad = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-10)
            x_adv = x_adv + alpha * grad
            # Project to L2 ball around x
            delta = x_adv - x
            delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
            factor = epsilon / (delta_norm + 1e-10)
            delta = delta * torch.clamp(factor.view(-1, 1, 1, 1), max=1.0)
            x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--max_images', type=int, default=3000)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ResNet50 pretrained on ImageNet-1k
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()

    # ImageNette val set (unnormalized for saving as uint8)
    data_path = os.path.join(DATA_DIR, 'val')
    raw_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # Normalized transform for model input
    norm_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(data_path, transform=raw_transform)
    dataset_norm = datasets.ImageFolder(data_path, transform=norm_transform)
    class_names = dataset.classes
    print(f"Classes: {class_names}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    loader_norm = torch.utils.data.DataLoader(dataset_norm, batch_size=args.batch_size, shuffle=False)

    all_images = []
    all_labels = []
    adv_images = {eps: [] for eps in args.epsilons}

    count = 0
    for (x_raw, y), (x_norm, _) in tqdm(zip(loader, loader_norm), total=len(loader), desc="L2 attack"):
        if count >= args.max_images:
            break

        x_norm = x_norm.to(device)
        y = y.to(device)

        # Save raw benign images as uint8
        x_uint8 = (x_raw.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        all_images.append(x_uint8)
        all_labels.append(y.cpu().numpy())

        # Generate L2 adversarial examples at different epsilons
        for eps in args.epsilons:
            x_adv_norm = pgd_l2(model, x_norm, y, epsilon=eps, steps=args.steps)
            # Denormalize back to uint8
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            x_adv_unnorm = x_adv_norm * std + mean
            x_adv_uint8 = (x_adv_unnorm.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            adv_images[eps].append(x_adv_uint8)

        count += x_raw.size(0)

    # Concatenate and save
    all_images = np.concatenate(all_images)[:args.max_images]
    all_labels = np.concatenate(all_labels)[:args.max_images]
    print(f"\nTotal images: {len(all_images)}")

    with open(os.path.join(OUTPUT_DIR, 'benign_imagenette.pkl'), 'wb') as f:
        pickle.dump({'images': all_images, 'labels': all_labels}, f)
    print(f"Saved benign: {all_images.shape}")

    for eps in args.epsilons:
        adv_images[eps] = np.concatenate(adv_images[eps])[:args.max_images]
        path = os.path.join(OUTPUT_DIR, f'adv_l2_eps{eps}.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'images': adv_images[eps], 'labels': all_labels, 'epsilon': eps}, f)
        print(f"Saved eps={eps}: {adv_images[eps].shape}")

    # Quick verification: check attack success rate
    print("\nAttack success rate (top-1 prediction changed):")
    for eps in args.epsilons:
        # Sample a small batch for quick verification
        sample_idx = np.random.choice(len(all_images), min(500, len(all_images)), replace=False)
        correct_clean = 0
        correct_adv = 0
        for idx in tqdm(sample_idx, desc=f"eps={eps}", leave=False):
            img_t = norm_transform(transforms.ToPILImage()(all_images[idx])).unsqueeze(0).to(device)
            adv_t = norm_transform(transforms.ToPILImage()(adv_images[eps][idx])).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_clean = model(img_t).argmax().item()
                pred_adv = model(adv_t).argmax().item()
            correct_clean += 1  # just count
            correct_adv += pred_clean == pred_adv
        print(f"  eps={eps}: pred_changed_rate={1 - correct_adv/len(sample_idx):.3f}")


if __name__ == '__main__':
    main()
