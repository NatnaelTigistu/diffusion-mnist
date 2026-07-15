import argparse
import copy

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import SimpleUNet
from diffusion import Diffusion


class EMA:
    """Exponential moving average of model weights. Sample from `ema_model`
    at inference time for noticeably cleaner generations than the raw
    (noisier) training weights.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay)


def main():
    parser = argparse.ArgumentParser()
    # Same flags as before, just better defaults for a diffusion model.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--out", type=str, default="../checkpoints/model.pt")
    # New, optional knobs — all default to sensible values, existing
    # callers that don't pass them get the improvements automatically.
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    use_amp = (device == "cuda") and not args.no_amp

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),  # rescale [0,1] -> [-1,1]
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = SimpleUNet().to(device)
    diffusion = Diffusion(timesteps=args.timesteps, device=device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(model, decay=args.ema_decay)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader)
        running_loss, n_batches = 0.0, 0

        for x, _ in pbar:
            x = x.to(device)
            t = torch.randint(0, args.timesteps, (x.shape[0],), device=device).long()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = diffusion.p_losses(model, x, t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            running_loss += loss.item()
            n_batches += 1
            pbar.set_description(f"epoch {epoch+1}/{args.epochs} | loss {loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / max(n_batches, 1)

        # Save the best checkpoint (by average epoch loss) instead of
        # blindly overwriting after every epoch. Save EMA weights, since
        # those are what you want to sample from.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ema.ema_model.state_dict(), args.out)
            print(f"epoch {epoch+1}: new best loss {best_loss:.4f}, saved to {args.out}")

    print(f"Done. Best model (EMA weights) saved to {args.out}")


if __name__ == "__main__":
    main()