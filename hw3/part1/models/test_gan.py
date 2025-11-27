import torch
from gan import DCGAN  # replace with the actual module name

# Small DCGAN config
dcgan_cfg = {"small": {"base_channels": 64, "latent_dim": 128, "epochs": 30, "lr": 2e-4}}

# Instantiate DCGAN
dcgan = DCGAN(
    image_channels=3,
    latent_dim=dcgan_cfg["small"]["latent_dim"],
    base_channels=dcgan_cfg["small"]["base_channels"],
)

# Create a dummy batch of real images
batch_size = 2
image_size = 32  # typical DCGAN image size
images = torch.randn(batch_size, 3, image_size, image_size)
batch = {"images": images}

# Forward pass
outputs = dcgan.forward(batch)

# Check outputs
assert "loss" in outputs, "Output missing 'loss'"
assert "generator_loss" in outputs, "Output missing 'generator_loss'"
assert "discriminator_loss" in outputs, "Output missing 'discriminator_loss'"
assert "fake_images" in outputs, "Output missing 'fake_images'"

# Check types
assert torch.is_tensor(outputs["loss"]), "loss is not a tensor"
assert torch.is_tensor(outputs["generator_loss"]), "generator_loss is not a tensor"
assert torch.is_tensor(outputs["discriminator_loss"]), "discriminator_loss is not a tensor"
assert torch.is_tensor(outputs["fake_images"]), "fake_images is not a tensor"

# Check shapes
assert outputs["fake_images"].shape == (batch_size, 3, image_size, image_size), \
    f"fake_images shape incorrect: {outputs['fake_images'].shape}"

# Check that losses are scalars
assert outputs["loss"].ndim == 0, f"loss is not a scalar: {outputs['loss'].shape}"
assert outputs["generator_loss"].ndim == 0, f"generator_loss is not a scalar: {outputs['generator_loss'].shape}"
assert outputs["discriminator_loss"].ndim == 0, f"discriminator_loss is not a scalar: {outputs['discriminator_loss'].shape}"

# Check fake image range (assuming Tanh)
assert outputs["fake_images"].min() >= -1.0 and outputs["fake_images"].max() <= 1.0, \
    "fake_images not in [-1, 1] range"

print("DCGAN forward test passed!")
