import argparse
import torch
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import SimpleUNet
from diffusion import Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--out", type=str, default="../checkpoints/model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),  # rescale [0,1] -> [-1,1]
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = SimpleUNet().to(device)
    diffusion = Diffusion(timesteps=args.timesteps, device=device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        pbar = tqdm(loader)
        for x, _ in pbar:
            x = x.to(device)
            t = torch.randint(0, args.timesteps, (x.shape[0],), device=device).long()

            optimizer.zero_grad()
            loss = diffusion.p_losses(model, x, t)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"epoch {epoch+1}/{args.epochs} | loss {loss.item():.4f}")

        torch.save(model.state_dict(), args.out)

    print(f"Done. Model saved to {args.out}")


if __name__ == "__main__":
    main()
