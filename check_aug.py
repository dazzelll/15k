from src.dataset import AIGCFolderDataset
import matplotlib.pyplot as plt

ds = AIGCFolderDataset("data/train_small", image_size=224, train=True, protocol_aug_prob=1.0)

n = 50
fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
for i in range(n):
    x_clean, x_t, y, severity, path = ds[i]
    axes[0, i].imshow(x_clean.permute(1, 2, 0))
    axes[0, i].set_title(f"clean, y={int(y)}", fontsize=9)
    axes[0, i].axis("off")
    axes[1, i].imshow(x_t.permute(1, 2, 0).clamp(0, 1))
    axes[1, i].set_title(f"sev={severity:.3f}", fontsize=9)
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("aug_check.png", dpi=120)
print("saved aug_check.png")