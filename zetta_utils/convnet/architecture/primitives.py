import torch
from zetta_utils import builder

# Sequence
builder.register("Sequential")(torch.nn.Sequential)

# Activation
builder.register("LeakyReLU")(torch.nn.LeakyReLU)
builder.register("ReLU")(torch.nn.ReLU)
builder.register("ELU")(torch.nn.ELU)
builder.register("Tanh")(torch.nn.ReLU)
builder.register("Sigmoid")(torch.nn.Sigmoid)
builder.register("Hardsigmoid")(torch.nn.Hardsigmoid)
builder.register("LogSigmoid")(torch.nn.LogSigmoid)
builder.register("LogSoftmax")(torch.nn.LogSoftmax)

# Normalization
builder.register("BatchNorm2d")(torch.nn.BatchNorm2d)
builder.register("BatchNorm3d")(torch.nn.BatchNorm3d)
builder.register("InstanceNorm2d")(torch.nn.InstanceNorm2d)
builder.register("InstanceNorm3d")(torch.nn.InstanceNorm3d)
# need num_channels to go first to be compatible with batchnorm
@builder.register("GroupNorm")
def compatible_group_norm(
    num_channels, num_groups, eps=1e-05, affine=True
) -> torch.nn.GroupNorm:  # pragma: no cover
    return torch.nn.GroupNorm(num_groups, num_channels, eps, affine)


# Convolutions
builder.register("Conv2d")(torch.nn.Conv2d)
builder.register("Conv3d")(torch.nn.Conv3d)
builder.register("ConvTranspose2d")(torch.nn.ConvTranspose2d)
builder.register("ConvTranspose3d")(torch.nn.ConvTranspose3d)

# Interpolation
# Can be done via zu.tensor_ops.interpolate
