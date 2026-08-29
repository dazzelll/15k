from src.dataset import AIGCFolderDataset
import matplotlib.pyplot as plt

ds = AIGCFolderDataset("data/train_small", image_size=224, train=True, protocol_aug_prob=1.0)

n = 12
fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

# AIGCFolderDataset.__getitem__ only returns severity, not the pipeline name,
# so call the underlying protocol transform directly here to see both.
for i in range(n):
    path, label = ds.samples[i]
    from PIL import Image
    img = Image.open(path).convert("RGB")
    x_clean = ds.to_tensor(img)
    img_t, name, severity = ds.protocol(img)
    x_t = ds.to_tensor(img_t)

    axes[0, i].imshow(x_clean.permute(1, 2, 0))
    axes[0, i].set_title(f"clean, y={label}", fontsize=8)
    axes[0, i].axis("off")

    axes[1, i].imshow(x_t.permute(1, 2, 0).clamp(0, 1))
    axes[1, i].set_title(f"{name}\nsev={severity:.3f}", fontsize=8)
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("aug_check.png", dpi=120)
print("saved aug_check.png")

# Also print a frequency count over more samples so you can see the actual
# pipeline distribution, not just what's visible in the 12-image grid.
from collections import Counter
counts = Counter()
n_samples = 300
for i in range(n_samples):
    path, _ = ds.samples[i % len(ds.samples)]
    img = Image.open(path).convert("RGB")
    _, name, _ = ds.protocol(img)
    counts[name] += 1

print(f"\nPipeline frequency over {n_samples} samples:")
for name, count in counts.most_common():
    print(f"  {name}: {count} ({100*count/n_samples:.1f}%)")