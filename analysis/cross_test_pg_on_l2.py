"""
Cross-test 2: PatchGuard (mask_bn) against L2 adversarial examples.
Tests each L2 epsilon against PatchGuard's provable masking at different patch_size settings.
"""
import sys, os
sys.path.insert(0, 'PatchGuard')
sys.path.insert(0, 'PatchGuard/nets')
sys.path.insert(0, 'PatchGuard/utils')

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pickle
from math import ceil
from tqdm import tqdm
import nets.bagnet
from utils.defense_utils import masking_defense, provable_masking

def main():
    device = 'cuda'
    rf_size = 17
    rf_stride = 8
    patch_sizes = [16, 32, 48]

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Load BagNet17
    model = nets.bagnet.bagnet17(pretrained=True, clip_range=None, aggregation='none')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = torch.nn.DataParallel(model)
    ckpt = torch.load('PatchGuard/checkpoints/bagnet17_nette.pth', map_location='cuda')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    epsilons = [0.5, 1.0, 2.0, 4.0]
    results = {}

    # Load benign data for natural accuracy baseline
    with open('shared_data/benign_imagenette.pkl', 'rb') as f:
        ben_data = pickle.load(f)

    for eps in epsilons:
        with open(f'shared_data/adv_l2_eps{eps}.pkl', 'rb') as f:
            adv_data = pickle.load(f)

        N = min(500, len(adv_data['images']))
        print(f"\n=== L2 eps={eps}, N={N} ===")

        for ps in patch_sizes:
            window_size = ceil((ps + rf_size - 1) / rf_stride)

            clean_corr = 0
            natural_corr = 0
            robust_cnt = 0
            vulnerable = 0
            incorrect = 0

            for i in tqdm(range(N), desc=f"ps={ps}", leave=False):
                img = adv_data['images'][i]
                label = adv_data['labels'][i]
                img_t = normalize(torch.tensor(img / 255.).permute(2, 0, 1).float()).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(img_t).detach().cpu().numpy()

                local_feat = out[0]
                natural_pred = np.argmax(np.mean(local_feat, axis=(0, 1)))
                natural_corr += natural_pred == label

                clean_pred = masking_defense(local_feat, window_shape=[window_size, window_size])
                result = provable_masking(local_feat, label, window_shape=[window_size, window_size])

                clean_corr += clean_pred == label
                if result == 0:
                    incorrect += 1
                elif result == 1:
                    vulnerable += 1
                else:
                    robust_cnt += 1

            key = f"eps{eps}_ps{ps}"
            results[key] = {
                'natural_acc': natural_corr / N,
                'defended_acc': clean_corr / N,
                'provable_robust': robust_cnt / N,
                'vulnerable': vulnerable / N,
                'incorrect': incorrect / N,
            }
            print(f"  ps={ps} (w={window_size}): natural={results[key]['natural_acc']:.3f}, "
                  f"defended={results[key]['defended_acc']:.3f}, robust={results[key]['provable_robust']:.3f}")

    with open('shared_data/cross_test_pg_l2.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nDone!")


if __name__ == '__main__':
    main()
