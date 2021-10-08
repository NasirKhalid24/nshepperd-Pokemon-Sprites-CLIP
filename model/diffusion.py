from .diffusion_utils import *

## Define the model (a residual U-Net)
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.class_embed = nn.Embedding(10, 4)

        self.net = nn.Sequential(
            ResConvBlock(3 + 16 + 4, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.image.Downsample2d(),  # 64x64 -> 32x32
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.image.Downsample2d(),  # 32x32 -> 16x16
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    SkipBlock([
                        nn.image.Downsample2d(),  # 16x16 -> 8x8
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        nn.image.Upsample2d(),
                    ]),
                    ResConvBlock(c * 4, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    nn.image.Upsample2d(),
                ]),
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.image.Upsample2d(),            # Haven't implemented ConvTranpose2d yet.
            ]),
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout=False),
        )

    def forward(self, cx, input, log_snrs, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(cx, log_snrs[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cx, cond), input.shape)
        return self.net(cx, jnp.concatenate([input, class_embed, timestep_embed], axis=1))