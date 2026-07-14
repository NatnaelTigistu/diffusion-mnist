import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from unet import SimpleUNet
from diffusion import Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="../checkpoints/model.pt")
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--out", type=str, default="../samples/grid.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    diffusion = Diffusion(timesteps=args.timesteps, device=device)
    final = diffusion.sample(model, image_size=28, batch_size=args.n, channels=1)

    final = ((final + 1) / 2).clamp(0, 1)  # [-1,1] -> [0,1]
    grid = make_grid(final, nrow=4)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved samples to {args.out}")


if __name__ == "__main__":
    main()
