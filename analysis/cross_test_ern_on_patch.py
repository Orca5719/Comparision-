"""
Cross-test 1: Erase-and-Restore detectors (trained on L2 adv) tested on patch attacks.
"""
import numpy as np
import pickle
import sys
sys.path.insert(0, 'Erase-and-Restore/erase_restore')
from ern_imagenette import load_model, extract_features

# Load model
import torch
device = torch.device('cuda')
model = load_model(device)

# Load patch attack data
patch_sizes = [16, 32, 48]
results = {}

for ps in patch_sizes:
    print(f"\n=== Patch size {ps}x{ps} ===")
    with open(f'shared_data/adv_patch{ps}.pkl', 'rb') as f:
        data = pickle.load(f)
    imgs = data['images'][:500]
    print(f"  {len(imgs)} images")

    print("  Extracting E&R features...")
    feats = extract_features(imgs, model, device, batch_size=32)
    print(f"  Features: {feats.shape}")

    # Test against all L2-trained detectors
    for eps in [0.5, 1.0, 2.0, 4.0]:
        with open(f'shared_data/results_ern/detectors_eps{eps}.pkl', 'rb') as f:
            dets = pickle.load(f)
        for clf_name in ['adaboost', 'svm']:
            clf = dets[clf_name]['model']
            pred = clf.predict(feats)
            det_rate = pred.mean()  # % classified as adversarial
            key = f"patch{ps}_detector_eps{eps}"
            results[key] = {clf_name: det_rate}
            print(f"  vs E&R(eps={eps}) {clf_name}: {det_rate:.3f} detected as adv")

# Save
with open('shared_data/results_ern/cross_test_patch.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nDone!")
