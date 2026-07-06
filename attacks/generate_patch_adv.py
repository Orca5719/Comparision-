"""Generate patch adversarial images by placing random colored squares."""
import pickle, numpy as np
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
ds = datasets.ImageFolder('PatchGuard/data/imagenette/val', transform=transform)

n = min(3000, len(ds))
images = np.zeros((n, 224, 224, 3), dtype=np.uint8)
labels = np.zeros(n, dtype=np.int64)

print(f'Loading {n} images...')
for i in range(n):
    img, labels[i] = ds[i]
    images[i] = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

for ps in [16, 32, 48]:
    adv = images.copy()
    print(f'Generating {ps}x{ps} patches...')
    for i in range(n):
        px = np.random.randint(0, 224 - ps)
        py = np.random.randint(0, 224 - ps)
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
        adv[i, px:px + ps, py:py + ps, :] = color
    path = f'shared_data/adv_patch{ps}.pkl'
    with open(path, 'wb') as f:
        pickle.dump({'images': adv, 'labels': labels, 'patch_size': ps}, f)
    print(f'  saved {path}')

print('Done!')
