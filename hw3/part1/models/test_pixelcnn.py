import torch
import torch.nn as nn
from pixelcnn import MaskedConv2d
# MaskedConv2d definition
# class MaskedConv2d(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, mask_type, **kwargs):
#         super().__init__(in_channels, out_channels, kernel_size, **kwargs)
#         self.mask_type = mask_type
#         self.register_buffer("mask", torch.ones_like(self.weight))
#         self.mask = self._build_mask()

#     def _build_mask(self):
#         mask = torch.ones_like(self.weight)
#         kh, kw = self.weight.shape[2], self.weight.shape[3]
#         y_c, x_c = kh // 2, kw // 2

#         # Zero out rows below the center
#         mask[:, :, y_c+1:, :] = 0

#         # Zero out columns strictly to the right in the center row
#         mask[:, :, y_c, x_c+1:] = 0

#         # Type-A also zeroes the center pixel
#         if self.mask_type == "A":
#             mask[:, :, y_c, x_c] = 0

#         return mask

# List of test shapes: (out_channels, in_channels, kh, kw)
test_shapes = [
    (64, 3, 7, 7),
    (64, 64, 3, 3),
    (64, 64, 3, 3),
    (64, 64, 3, 3),
    (64, 64, 3, 3),
    (64, 64, 3, 3),
    (64, 64, 1, 1),
    (3, 3, 3, 3),
    (3, 3, 3, 3),
    (64, 3, 7, 7),
    (64, 64, 3, 3),
    (64, 64, 3, 3),
    (64, 64, 3, 3),
    (64, 64, 3, 3),
    (64, 64, 3, 3),
    (64, 64, 1, 1),
]

def test_masked_conv_shapes(mask_type):
    for idx, (out_ch, in_ch, kh, kw) in enumerate(test_shapes):
        conv = MaskedConv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(kh, kw),
            mask_type=mask_type,
            padding=(kh//2, kw//2),
            bias=True,
        )

        # Check center coordinates
        y_c, x_c = kh // 2, kw // 2

        # Pick first out_channel and first in_channel for testing
        mask_slice = conv.mask[0, 0]

        # Rows below center
        rows_below = mask_slice[y_c+1:, :]
        assert torch.all(rows_below == 0), f"[{idx}] Rows below center not zeroed for {mask_type}"

        # Columns to the right in center row
        center_row_right = mask_slice[y_c, x_c+1:]
        assert torch.all(center_row_right == 0), f"[{idx}] Pixels to the right not zeroed for {mask_type}"

        # Center pixel

        center_pixel = mask_slice[y_c, x_c]
        if mask_type == "A":
            assert center_pixel == 0, f"[{idx}] Center pixel not zeroed for type-A"
        else:
            assert center_pixel == 1, f"[{idx}] Center pixel should not be zeroed for type-B"

        print(f"[{idx}] Mask type-{mask_type} shape {conv.mask.shape} passed")

# Run tests
print("Testing Type-A masks")
test_masked_conv_shapes("A")

print("\nTesting Type-B masks")
test_masked_conv_shapes("B")
