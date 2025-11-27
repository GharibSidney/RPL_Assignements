import torch
from torch import nn
import torch.nn.functional as F

### WARNING: DO NOT EDIT THE CLASS NAME, INITIALIZER, AND GIVEN INPUTS AND ATTRIBUTES. OTHERWISE, YOUR TEST CASES CAN FAIL. ###


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.mask_type = mask_type
        self.register_buffer("mask", torch.ones_like(self.weight))
        self.mask = self._build_mask()

    def _build_mask(self):
        mask = torch.ones_like(self.weight)
        print("Mask shape", mask.shape)
        # Mask type A blocks the current pixel in addition to future pixels; type B allows the current pixel.
        kernel_height, kernel_width = self.weight.shape[2], self.weight.shape[3]
        y_center, x_center = kernel_height // 2, kernel_width // 2
        # TODO: zero out rows strictly below the current pixel
        mask[:, :, y_center+1:, :] = 0
        # TODO: zero out columns to the right of the current pixel (respecting mask type)
        mask[:, :, y_center, x_center+1:] = 0
        if self.mask_type == "A":
            # TODO: zero the current pixel for type A masks
            mask[:, :, y_center, x_center] = 0
        return mask

    def forward(self, x):
        weight = self.weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResidualConnection(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class PixelCNN(nn.Module):
    def __init__(self, image_channels=3, hidden_channels=64, residual_layers=5, bins=256):
        super().__init__()
        self.conv_cls = MaskedConv2d
        self.image_channels = image_channels
        self.bins = bins
        self.net = self._build_network(image_channels, hidden_channels, residual_layers, bins)

    def _residual_block(self, channels):
        block = ResidualConnection(
            nn.Sequential(
                nn.ReLU(),
                self.conv_cls(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    mask_type="B",
                    bias=True,
                ),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
        )
        return block

    def _build_network(self, image_channels, hidden_channels, residual_layers, bins):
        layers = [
            self.conv_cls(
                image_channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
                mask_type="A",
                bias=True,
            ),
            nn.ReLU(),
        ]
        for _ in range(residual_layers):
            layers.append(self._residual_block(hidden_channels))
        layers.extend(
            [
                nn.ReLU(),
                self.conv_cls(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=1,
                    padding=0,
                    mask_type="B",
                    bias=True,
                ),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, image_channels * bins, kernel_size=1),
            ]
        )
        network = nn.Sequential(*layers)
        return network

    def forward(self, batch):
        x = batch["images"]
        logits = self.net(x)  # TODO: compute logits over discrete bins
        b, c_bins, h, w = logits.shape
        logits = logits.view(b, self.image_channels, self.bins, h, w)
        targets = batch.get("targets")
        if targets is None:
            targets = torch.clamp((x * (self.bins - 1)).long(), 0, self.bins - 1)#...  # TODO: quantise the input images to bin indices
        else:
            targets = targets.long()
        logits_flat = logits.permute(0, 3, 4, 1, 2).reshape(-1, self.bins)
        targets_flat = targets.permute(0, 2, 3, 1).reshape(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)  # TODO: compute the cross-entropy loss
        logits = logits.view(b, self.image_channels * self.bins, h, w)
        return {
            "loss": loss,
            "logits": logits,
            "targets": targets,
        }