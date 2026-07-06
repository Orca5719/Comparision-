"""
Erase-and-Restore adapted for ImageNette (10-class), PyTorch GPU version.
Ref: "Exploiting the Sensitivity of L2 Adversarial Examples to Erase-and-Restore" (AsiaCCS'21)

Feature: 12 predictions (1 clean + 11 inpaint) × 10 classes = 120D.
"""
import numpy as np
import pickle
import copy
import cv2
import os
import random
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from skimage import img_as_ubyte
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# ImageNet 1000-class indices for the 10 ImageNette classes
IMAGENETTE_IDX = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

IMG_SIZE = 224
TRY_TIMES = 11
NUM_MASK_PIXELS = 300

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ===== Random mask generation (unchanged from original) =====

def get_single_mask(num):
    x_b = np.zeros((IMG_SIZE, IMG_SIZE))
    for _ in range(num):
        x_axis = random.randint(1, 219)
        y_axis = random.randint(1, 218)
        case = random.randint(1, 5)
        if case == 1:
            x_b[x_axis][y_axis] = 1
            x_b[x_axis][y_axis+1] = 1
            x_b[x_axis][y_axis+2] = 1
            x_b[x_axis][y_axis+3] = 1
            x_b[x_axis][y_axis+4] = 1
            x_b[x_axis+1][y_axis] = 1
            x_b[x_axis+1][y_axis+1] = 1
            x_b[x_axis+1][y_axis+2] = 1
            x_b[x_axis+1][y_axis+3] = 1
            x_b[x_axis+1][y_axis+4] = 1
        elif case == 2:
            x_b[x_axis][y_axis] = 1
            x_b[x_axis][y_axis+1] = 1
            x_b[x_axis][y_axis+2] = 1
            x_b[x_axis+1][y_axis] = 1
            x_b[x_axis+1][y_axis+1] = 1
            x_b[x_axis+1][y_axis+2] = 1
            x_b[x_axis+1][y_axis+3] = 1
            x_b[x_axis+1][y_axis+4] = 1
            x_b[x_axis+2][y_axis+3] = 1
            x_b[x_axis+2][y_axis+4] = 1
        elif case == 3:
            x_b[x_axis][y_axis] = 1
            x_b[x_axis][y_axis+1] = 1
            x_b[x_axis+1][y_axis] = 1
            x_b[x_axis+1][y_axis+1] = 1
            x_b[x_axis+1][y_axis+2] = 1
            x_b[x_axis+1][y_axis+3] = 1
            x_b[x_axis+1][y_axis+4] = 1
            x_b[x_axis+2][y_axis+2] = 1
            x_b[x_axis+2][y_axis+3] = 1
            x_b[x_axis+2][y_axis+4] = 1
        elif case == 4:
            x_b[x_axis][y_axis] = 1
            x_b[x_axis+1][y_axis] = 1
            x_b[x_axis+1][y_axis+1] = 1
            x_b[x_axis+1][y_axis+2] = 1
            x_b[x_axis+1][y_axis+3] = 1
            x_b[x_axis+1][y_axis+4] = 1
            x_b[x_axis+2][y_axis+1] = 1
            x_b[x_axis+2][y_axis+2] = 1
            x_b[x_axis+2][y_axis+3] = 1
            x_b[x_axis+2][y_axis+4] = 1
        elif case == 5:
            x_b[x_axis][y_axis] = 1
            x_b[x_axis][y_axis+1] = 1
            x_b[x_axis][y_axis+2] = 1
            x_b[x_axis][y_axis+3] = 1
            x_b[x_axis+1][y_axis] = 1
            x_b[x_axis+1][y_axis+1] = 1
            x_b[x_axis+1][y_axis+2] = 1
            x_b[x_axis+1][y_axis+3] = 1
            x_b[x_axis+1][y_axis+4] = 1
            x_b[x_axis+2][y_axis+4] = 1
    return img_as_ubyte(x_b)


def get_masks(num, try_times=TRY_TIMES):
    mask = np.expand_dims(get_single_mask(num), axis=0)
    for _ in range(try_times - 1):
        m = np.expand_dims(get_single_mask(num), axis=0)
        mask = np.concatenate((mask, m))
    return mask  # (11, 224, 224)


MASK = get_masks(NUM_MASK_PIXELS)


# ===== Model =====

class ImageNetteResNet(nn.Module):
    """ResNet50 pretrained on ImageNet-1k, outputs only 10 ImageNette class logits."""
    def __init__(self):
        super().__init__()
        full = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(full.children())[:-1])  # up to avgpool
        self.fc = full.fc  # (1000,) linear layer

    def forward(self, x):
        x = self.backbone(x)  # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (B, 2048)
        x = self.fc(x)  # (B, 1000)
        return x[:, IMAGENETTE_IDX]  # (B, 10)


def load_model(device):
    model = ImageNetteResNet().to(device)
    model.eval()
    return model


# ===== Preprocessing =====

normalize = transforms.Normalize(mean=MEAN, std=STD)


def img_to_tensor(img_np):
    """uint8 (H,W,3) 0-255 → normalized tensor (3,H,W)."""
    t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    return normalize(t)


# ===== Batch Erase-and-Restore =====

def extract_features(images, model, device, batch_size=32):
    """
    images: numpy (N, H, W, 3) uint8
    Returns: (N, 120) feature matrix
    """
    N = len(images)
    all_preds = []  # list of (N, 10) arrays

    # Helper: batch inference
    def batch_predict(img_list):
        preds = []
        for i in range(0, len(img_list), batch_size):
            batch = img_list[i:i+batch_size]
            tensors = [img_to_tensor(b).to(device) for b in batch]
            x = torch.stack(tensors)
            with torch.no_grad():
                y = model(x).cpu().numpy()
            preds.append(y)
        return np.concatenate(preds, axis=0)

    # 1. Clean predictions
    all_preds.append(batch_predict([images[i] for i in range(N)]))

    # 2-12. Inpaint predictions for each mask
    for k in range(TRY_TIMES):
        inpainted = []
        for i in range(N):
            src = np.ubyte(images[i])
            dst = cv2.inpaint(src, MASK[k], 3, cv2.INPAINT_TELEA)
            inpainted.append(dst)
        all_preds.append(batch_predict(inpainted))
        if (k + 1) % 4 == 0:
            print(f"  mask {k+1}/{TRY_TIMES} done")

    # Stack: (12, N, 10) → (N, 12, 10) → (N, 120)
    stacked = np.stack(all_preds, axis=1)  # (N, 12, 10)
    return stacked.reshape(N, -1)


# ===== Detector training =====

def train_detectors(X_train, y_train, X_test, y_test):
    results = {}
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
    ada.fit(X_train, y_train)
    results['adaboost'] = {
        'model': ada,
        'train_acc': ada.score(X_train, y_train),
        'test_acc': ada.score(X_test, y_test),
        'test_auc': roc_auc_score(y_test, ada.predict_proba(X_test)[:, 1])
    }
    svc = SVC(kernel='rbf', gamma=0.01, C=5, probability=True)
    svc.fit(X_train, y_train)
    results['svm'] = {
        'model': svc,
        'train_acc': svc.score(X_train, y_train),
        'test_acc': svc.score(X_test, y_test),
        'test_auc': roc_auc_score(y_test, svc.predict_proba(X_test)[:, 1])
    }
    return results


# ===== Main =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benign_pkl', default='shared_data/benign_imagenette.pkl')
    parser.add_argument('--adv_pkl_pattern', default='shared_data/adv_l2_eps{eps}.pkl')
    parser.add_argument('--output_dir', default='shared_data/results_ern')
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(device)

    # Load data
    print("Loading benign images...")
    with open(args.benign_pkl, 'rb') as f:
        ben_data = pickle.load(f)
    ben_imgs = ben_data['images'][:args.n_samples]
    print(f"  {len(ben_imgs)} benign images")

    # Extract benign features
    print("Extracting benign features...")
    ben_feat = extract_features(ben_imgs, model, device, args.batch_size)
    feat_path = os.path.join(args.output_dir, 'ben_feat_120.npy')
    np.save(feat_path, ben_feat)
    print(f"  saved {ben_feat.shape}")

    for eps in args.epsilons:
        adv_path = args.adv_pkl_pattern.format(eps=eps)
        print(f"\n=== Epsilon={eps} ===")
        print(f"Loading {adv_path}...")
        with open(adv_path, 'rb') as f:
            adv_data = pickle.load(f)
        adv_imgs = adv_data['images'][:args.n_samples]

        print(f"Extracting adv features...")
        adv_feat = extract_features(adv_imgs, model, device, args.batch_size)
        np.save(os.path.join(args.output_dir, f'adv_feat_eps{eps}_120.npy'), adv_feat)

        n = min(len(ben_feat), len(adv_feat))
        split = int(n * 0.8)
        X = np.concatenate([ben_feat[:n], adv_feat[:n]])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        idx = np.random.RandomState(42).permutation(len(X))
        X, y = X[idx], y[idx]
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        results = train_detectors(X_train, y_train, X_test, y_test)
        for name, res in results.items():
            print(f"  {name}: train_acc={res['train_acc']:.4f}, "
                  f"test_acc={res['test_acc']:.4f}, test_auc={res['test_auc']:.4f}")

        with open(os.path.join(args.output_dir, f'detectors_eps{eps}.pkl'), 'wb') as f:
            pickle.dump(results, f)

    print("\n=== Erase-and-Restore pipeline complete ===")


if __name__ == '__main__':
    main()
