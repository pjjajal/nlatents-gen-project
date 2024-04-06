import torch
import torch.nn as nn
import torch.nn.functional as F


class Embed(nn.Module):
    def __init__(
        self,
        channels,
        embed_dim,
        channel_kernel_size = None,
    ) -> None:
        super().__init__()
        self.channel_kernel_size = channel_kernel_size
        if channel_kernel_size is not None:
            padding = (channels % channel_kernel_size) // 2
            self.proj = nn.Conv2d(
                1,
                embed_dim,
                kernel_size=(1, channel_kernel_size),
                stride=(1, channel_kernel_size),
                padding=(0, padding),
            )
        else:
            self.proj = nn.Linear(channels, embed_dim)

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time, channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channel_patches, time, embed_dim) or (batch_size, time, embed_dim).

        """
        if self.channel_kernel_size is not None:
            x = x.unsqueeze(1)
        x = self.proj(x)
        if self.channel_kernel_size is not None:
            x = x.permute(
                0, 3, 2, 1
            )  # (batch_size, embed_dim, time, channel_patches) -> (batch_size, channel_patches, time, embed_dim)
        return x


if __name__ == "__main__":
    patch_embed = Embed(137, 768, 75)
    x = torch.randn(1, 140, 137)
    y = patch_embed(x)
    print(y.shape)

    patch_embed = Embed(137, 768)
    x = torch.randn(1, 140, 137)
    y = patch_embed(x)
    print(y.shape)