model: {
    "@type": "ConvBlock"
    "@version": "0.0.1"
    num_channels: [16, 16, 16, 16, 16]
    kernel_sizes: [3, 3, 3]
    conv: {
        "@type": "torch.nn.Conv3d"
        "@mode": "partial"
        bias: false
    }
    activation: {
        "@type": "torch.nn.ReLU"
        "@mode": "partial"
        inplace: true
    }
    normalization: {
        "@type": "torch.nn.InstanceNorm3d"
        "@mode": "partial"
        affine: false
    }
    skips: {"1": 3}
    normalize_last: true
    activate_last: true
    paddings: [1, 1, 1]
    activation_mode: "post"
}
