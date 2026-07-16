import torch
from torchvision.utils import save_image
from unet import SimpleUNet
from diffusion import Diffusion

device = "cpu"
model = SimpleUNet().to(device)
model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device))
model.eval()

diffusion = Diffusion(timesteps=300, device=device)
samples = diffusion.sample(model, image_size=28, batch_size=16, channels=1)

# samples are typically in [-1, 1] since training normalized with (t*2)-1;
# rescale to [0, 1] for saving as a viewable image
samples = (samples.clamp(-1, 1) + 1) / 2

save_image(samples, "samples/output.png", nrow=4)
print("Saved to samples/output.png")