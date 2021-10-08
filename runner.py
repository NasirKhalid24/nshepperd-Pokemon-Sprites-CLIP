import sys
import time
import numpy as np
from tqdm import tqdm, trange
from functools import partial

import jax
from jax.config import config
import jax.numpy as jnp

sys.path.append('jaxtorch')
sys.path.append('CLIP_JAX')

import jaxtorch
from jaxtorch import PRNG, Context

import clip_jax

# For data loading.
from torchvision import  utils
from torchvision.transforms import functional as TF
import torch.utils.data
import torch

# Model
from model import Diffusion

# Helper functions
from utils import *

class StateDict(dict):
    pass

## Actually do the run
def demo(
    prompt="",
    clip_size=224,
    eta=1.0,
    clip_guidance_scale=2000,
    seed=0,
    steps=250,
    batch_size=16,
    image_size=64,
    FLOAT64=False):

    if FLOAT64:        
        config.update("jax_enable_x64", True)
        
    ## Load models
    print('Using device:', jax.devices())

    model = Diffusion()
    params_ema = model.init_weights(jax.random.PRNGKey(0))
    print('Model parameters:', sum(np.prod(p.shape) for p in params_ema.values.values()))

    # Load checkpoint
    
    state_dict = jaxtorch.pt.load(fetch_model('https://set.zlkj.in/models/diffusion/pokemon_diffusion_gen3+4_c64_6783.pth'))
    model.load_state_dict(params_ema, state_dict['model_ema'], strict=False)

    print('Loading CLIP model...')
    image_fn, text_fn, clip_params, _ = clip_jax.load('ViT-B/32')
    normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711])

    if FLOAT64:
        ## Convert params for both to float64
        params_ema = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params_ema)
        clip_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), clip_params)

    tqdm.write('Sampling...')
    rng = PRNG(jax.random.PRNGKey(seed))

    fakes = jax.random.normal(rng.split(), [batch_size, 3, image_size, image_size])

    fakes_classes = jnp.array([0] * batch_size) # plain
    ts = jnp.ones([batch_size])

    # Create the noise schedule
    t = jnp.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    ## Define model wrappers
    @jax.jit
    def eval_model(params, xs, ts, classes, key):
        cx = Context(params, key).eval_mode_()
        return model(cx, xs, ts, classes)

    def txt(prompt, text_fn):
        """Returns normalized embedding."""
        text = clip_jax.tokenize([prompt])
        text_embed = text_fn(clip_params, text)
        return norm1(text_embed.reshape(512))

    def emb_image(image, clip_params=None):
        return norm1(image_fn(clip_params, image))

    def base_cond_fn(x, t, text_embed, clip_guidance_scale, classes, key, params_ema, clip_params):
        rng = PRNG(key)
        n = x.shape[0]

        log_snrs = get_ddpm_schedule(t)
        alphas, sigmas = get_alphas_sigmas(log_snrs)

        def denoise(x, key):
            eps = eval_model(params_ema, x, log_snrs.broadcast_to([n]), classes, rng.split())
            # Predict the denoised image
            pred = (x - eps * sigmas) / alphas
            x_in = pred * sigmas + x * alphas
            return x_in
        
        x_in, backward = jax.vjp(partial(denoise, key=rng.split()), x)

        def clip_loss(x_in):
            x_in = jax.image.resize(x_in, [n, 3, 224, 224], method='nearest')
            clip_in = normalize(x_in.add(1).div(2))
            image_embeds = emb_image(clip_in, clip_params).reshape([n, 512])
            losses = spherical_dist_loss(image_embeds, text_embed)
            return losses.sum() * clip_guidance_scale
        clip_grad = jax.grad(clip_loss)(x_in)

        return -backward(clip_grad)[0]
    base_cond_fn = jax.jit(base_cond_fn)

    def cond_fn(*args, **kwargs):
        grad = base_cond_fn(*args, **kwargs)
        # Gradient nondeterminism monitoring
        # grad2 = base_cond_fn(*args, **kwargs)
        # average = (grad + grad2) / 2
        # print((grad - grad2).abs().mean() / average.abs().mean())
        return grad

    text_embed = txt(prompt, text_fn)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (eps, the predicted noise)
        eps = eval_model(params_ema, fakes, ts * log_snrs[i], fakes_classes, rng.split())

        # Predict the denoised image
        pred = (fakes - eps * sigmas[i]) / alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:

            cond_score = cond_fn(fakes, t[i], text_embed, clip_guidance_scale, fakes_classes, rng.split(), params_ema, clip_params)

            eps = eps - sigmas[i] * cond_score
            pred = (fakes - eps * sigmas[i]) / alphas[i]

            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            fakes = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                fakes += jax.random.normal(rng.split(), fakes.shape) * ddim_sigma

        # If we are on the last timestep, output the denoised image
        else:
            fakes = pred

    grid = utils.make_grid(torch.tensor(np.array(fakes)), 4).cpu()
    timestring = time.strftime('%Y%m%d%H%M%S')
    os.makedirs('samples', exist_ok=True)
    filename = f'samples/{timestring}_{prompt}.png'
    TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(filename)
    # display.display(display.Image(filename))
    print(f'Saved {filename}')