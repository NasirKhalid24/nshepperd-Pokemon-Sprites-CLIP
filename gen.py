"""
Written by @nshepperd1 
https://github.com/nshepperd

Note from original code:

"Generates pixel artwork from a prompt using a diffusion model trained on pokemon sprites. Thanks Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings) for the diffusion model design :)"
"""
from runner import demo

## Option, not sure if helpful: sample in float64 to fix nondeterminism of CLIP's backward
FLOAT64=False

## Settings for the run
seed = 0

# Prompt for CLIP guidance
prompt = "pikachu #pixelart"

# Strength of conditioning
clip_guidance_scale = 2000

# The amount of noise to add each timestep when sampling
# 0 = no noise (DDIM)
# 1 = full noise (DDPM)
eta = 1.0

batch_size = 16

# Image size. Was trained on 64x64. Must be a multiple of 8 but different sizes are possible.
image_size = 64

# Number of steps for sampling, more = better quality generally
steps = 250

class StateDict(dict):
    pass

demo()