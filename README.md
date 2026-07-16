**DDPM on MNIST**  
A minimal, from-scratch implementation of a Denoising Diffusion Probabilistic  
   
 Model (Ho et al., 2020), trained on MNIST.  
- Forward process: closed-form Gaussian noising, src/diffusion.py  
- Reverse process: small UNet predicts the noise added at each step, src/unet.py  
- Training objective: simplified noise-prediction MSE loss  
No local GPU needed — training runs on Google Colab's free GPU via the  
   
 notebook in notebooks/.  
**Project structure**  
diffusion-mnist/  
 ├── README.md  
 ├── requirements.txt  
 ├── .gitignore  
 ├── src/  
 │   ├── diffusion.py     # noise schedule, forward/reverse process, loss  
 │   ├── unet.py           # time-conditioned UNet  
 │   ├── train.py          # training loop  
 │   └── sample.py         # generate images from a trained model  
 ├── notebooks/  
 │   └── DDPM_MNIST_Colab.ipynb  
 ├── checkpoints/           # trained weights (gitignored)  
 └── samples/                # generated sample grids  
   
**Setup**  
   
**1. Train on Colab**  
1. Open notebooks/DDPM_MNIST_Colab.ipynb in Google Colab (upload it, or  
   
 open directly from GitHub via File -> Open notebook -> GitHub tab).  
2. Set the runtime to a GPU: **Runtime -> Change runtime type -> T4 GPU**.  
3. Edit the REPO_URL variable in the first cell to point at your repo.  
4. Run all cells. This clones your repo, installs dependencies, trains for  
   
 50 epochs (~20-30 min on a T4), and generates a sample grid.  
   
   
**3. Get your trained model/results back out**  
Easiest path: download checkpoints/model.pt and samples/grid.png  
   
 directly from the Colab file browser (left sidebar) and keep them locally,  
   
 or commit just the small sample image back to GitHub (last cell in the  
   
 notebook does this — you'll need a  
   
 [GitHub Personal Access Token since  
   
 Colab can't use your local git credentials).](https://github.com/settings/tokens "https://github.com/settings/tokens")  
Don't commit the checkpoint file itself — it's a few MB of binary weights,  
   
 which is what .gitignore is for. If you want it versioned, use  
   
 [Git LFS or just link to it in the README (e.g.  
   
 uploaded to Google Drive).](https://git-lfs.com/ "https://git-lfs.com/")  
   
**Results**  
After training, drop your sample grid here:  
samples/grid.png  
**References**  
- Ho, Jain, Abbeel — [Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/abs/2006.11239 "https://arxiv.org/abs/2006.11239")  
- Lilian Weng — [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/")  
