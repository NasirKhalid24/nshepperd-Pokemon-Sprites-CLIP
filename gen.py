"""
Written by @nshepperd1 
https://github.com/nshepperd

Note from original code:

"Generates pixel artwork from a prompt using a diffusion model trained on pokemon sprites. Thanks Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings) for the diffusion model design :)"
"""

import argparse
from runner import demo

parser = argparse.ArgumentParser(description='Written by nshepperd: "Generates pixel artwork from a prompt using a diffusion model trained on pokemon sprites. Thanks Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings) for the diffusion model design :)"')

parser.add_argument('--seed', type=int,
                    help='Seed for random number generator', default=0)

parser.add_argument('--float64', dest='float64', action='store_true', help='Sample in float64 to fix nondeterminism of CLIPs backward')
parser.add_argument('--no-float64', dest='float64', action='store_false')
parser.set_defaults(float64=False)

parser.add_argument('--prompt', type=str, help="Prompt for CLIP guidance", default="a blue fairy type pokemon with wings #pixelart")

parser.add_argument('--cgs', type=int, help="Strength of conditioning | factor multiplied by loss", default=2000)

def limit_eta(arg):
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < 0.0 or f > 1.0:
        raise argparse.ArgumentTypeError("Argument must be < " + str(1.0) + "and > " + str(0.0))
    return f
parser.add_argument('--eta', type=limit_eta, help="The amount of noise to add each timestep when sampling. 0 is none and 1 is max", default=1.0)

parser.add_argument('--batch_size', type=int, help="Batch Size (# of samples gen)", default=16)

def limit_size(arg):
    try:
        f = int(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a integer number")
    if f % 8 != 0:
        raise argparse.ArgumentTypeError("Must be a multiple of 8")
    if f < 8:
        raise argparse.ArgumentTypeError("Must be larger")
    return f
parser.add_argument('--image_size', type=limit_size, help="Image size. Was trained on 64x64. Must be a multiple of 8 but different sizes are possible.", default=64)

parser.add_argument('--steps', type=int, help="Number of steps for sampling, more = better quality generally", default=250)

args = parser.parse_args()

class StateDict(dict):
    pass

demo(
    seed=args.seed,
    FLOAT64=args.float64,
    prompt=args.prompt,
    eta=args.eta,
    clip_guidance_scale=args.cgs,
    batch_size=args.batch_size,
    image_size=args.image_size,
    steps=args.steps,
    clip_size=224)