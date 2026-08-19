#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""SUPPNet architecture — PyTorch implementation.

This is a faithful, weight-compatible port of the original Keras/TensorFlow
definition of SUPPNet.  It builds the very same computational graph and consumes
the *unmodified* Keras weight files shipped in ``supp_models_modernized``, so
predictions are numerically identical to the TensorFlow implementation (up to
float32 round-off).

The primitives in the first section reproduce a handful of TensorFlow-specific
conventions that PyTorch does not share out of the box:

* ``padding='same'`` on a strided ``Conv1D`` pads asymmetrically (extra column on
  the right), while PyTorch only supports symmetric padding.
* ``Conv1DTranspose(padding='same')`` crops the full transposed-convolution
  output starting from the *left* edge, whereas ``nn.ConvTranspose1d`` crops
  symmetrically via its ``padding`` argument.
* ``UpSampling2D(interpolation='bilinear')`` resolves to ``tf.image.resize``,
  i.e. half-pixel centres without corner alignment — the same convention as
  ``F.interpolate(..., mode='linear', align_corners=False)``.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .keras_weights import load_keras_weights

__all__ = [
    "SUPPNet",
    "create_SUPPNet_model",
    "get_suppnet_model",
    "modelWrapper",
    "ModelWrapper",
    "select_device",
    "default_cpu_threads",
    "WEIGHTS_FILES",
]


WEIGHTS_FILES = {
    "synth": "supp_models_modernized/SUPPNet_synth.weights.h5",
    "active": "supp_models_modernized/SUPPNet_active.weights.h5",
    "emission": "supp_models_modernized/SUPPNet_18_powr.weights.h5",
}

WEIGHTS_DESCRIPTIONS = {
    "synth": "SUPPNet (synth)",
    "active": "SUPPNet (active)",
    "emission": "SUPPNet (emission, active+PoWR)",
}

# The pyramid pooling factors are fixed when the graph is built, so the network
# only accepts windows of this length.
WINDOW_LENGTH = 8192

DEFAULT_PARAMS = {
    "d_i": np.array([1, 1, 1, 2, 2, 5, 6, 7, 10, 7, 6, 5, 2, 2, 1, 1, 1]),
    "w_i": np.array([12, 16, 16, 20, 24, 32, 44, 44, 44, 44, 44, 32, 24, 20, 16, 16, 12]),
    "psp_bool": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
    "g": 64,
    "w_ppm": 4,
    "d_ppm": 1,
}


# --------------------------------------------------------------------------- #
# TensorFlow-compatible primitives
# --------------------------------------------------------------------------- #


def _same_pad_total(length: int, kernel_size: int, stride: int) -> int:
    """Total padding TensorFlow inserts for ``padding='SAME'``."""
    out_length = -(-length // stride)
    return max((out_length - 1) * stride + kernel_size - length, 0)


class StridedConv1dSame(nn.Conv1d):
    """``Conv1D(..., strides>1, padding='same')`` with TensorFlow padding split.

    TensorFlow puts the smaller half of the padding on the left, which for the
    odd padding totals produced by strided convolutions differs from PyTorch's
    symmetric padding.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        total = _same_pad_total(x.shape[-1], self.kernel_size[0], self.stride[0])
        if total:
            x = F.pad(x, (total // 2, total - total // 2))
        return self._conv_forward(x, self.weight, self.bias)


class ConvTranspose1dSame(nn.ConvTranspose1d):
    """``Conv1DTranspose(..., padding='same')``.

    Keras sizes the output as ``input_length * stride`` and delegates to
    ``tf.nn.conv1d_transpose``, which is the gradient of a ``SAME``-padded
    forward convolution.  In practice that means the full transposed output is
    cropped starting at ``max(kernel_size - stride, 0) // 2`` (zero-padded on the
    right when the kernel is shorter than the stride).  The bias is applied after
    that resize, so it is kept out of the convolution itself.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, bias=False)
        self.tf_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        out_length = x.shape[-1] * self.stride[0]
        y = super().forward(x)
        full_length = y.shape[-1]
        if full_length > out_length:
            offset = (full_length - out_length) // 2
            y = y[..., offset:offset + out_length]
        elif full_length < out_length:
            y = F.pad(y, (0, out_length - full_length))
        return y + self.tf_bias.view(1, -1, 1)


def same_avg_pool1d(x, pool_size):
    """``AveragePooling1D(pool_size, pool_size, padding='same')``.

    TensorFlow excludes the implicit padding from the average, so the tail window
    is divided by the number of real samples it covers.
    """
    length = x.shape[-1]
    total = _same_pad_total(length, pool_size, pool_size)
    if not total:
        return F.avg_pool1d(x, pool_size, pool_size)
    left, right = total // 2, total - total // 2
    padded = F.pad(x, (left, right))
    sums = F.avg_pool1d(padded, pool_size, pool_size) * pool_size
    counts = F.avg_pool1d(
        F.pad(x.new_ones((1, 1, length)), (left, right)), pool_size, pool_size
    ) * pool_size
    return sums / counts


def upsample_linear(x, factor):
    """``UpSampling2D(size=(factor, 1), interpolation='bilinear')`` on (L, 1, C).

    Resizing a width-1 image only interpolates along the length axis, and
    ``tf.image.resize`` uses half-pixel centres — identical to PyTorch's
    ``align_corners=False`` linear interpolation.
    """
    return F.interpolate(x, size=x.shape[-1] * factor, mode="linear", align_corners=False)


# --------------------------------------------------------------------------- #
# Layer bookkeeping
# --------------------------------------------------------------------------- #


class KerasLayerOrder:
    """Records weighted layers in the order Keras would have created them.

    Keras derives automatic layer names from a per-class counter, so the *n*-th
    ``conv1d_*`` layer in a checkpoint corresponds to the *n*-th ``Conv1D``
    instantiated while building the model.  Recording that order here is what
    lets the original weight files be loaded without any renaming.
    """

    def __init__(self):
        self.conv_layers = []
        self.deconv_layers = []
        self.named_layers = {}

    def conv(self, in_channels, out_channels, kernel_size, stride=1, name=None):
        if stride == 1:
            # TensorFlow's 'same' padding is symmetric for odd kernels at stride 1.
            if kernel_size % 2 == 0:
                raise ValueError("Only odd kernel sizes are supported at stride 1.")
            layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        else:
            layer = StridedConv1dSame(in_channels, out_channels, kernel_size, stride)
        return self._register(layer, self.conv_layers, name)

    def deconv(self, in_channels, out_channels, kernel_size, stride, name=None):
        layer = ConvTranspose1dSame(in_channels, out_channels, kernel_size, stride)
        return self._register(layer, self.deconv_layers, name)

    def _register(self, layer, auto_named, name):
        if name is None:
            auto_named.append(layer)
        else:
            self.named_layers[name] = layer
        return layer


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


def _resolve_group_width(width, group_width):
    return width if group_width is None else min(group_width, width)


class ResidualBlock(nn.Module):
    """``residual_block`` — identity shortcut, so ``in_channels`` must equal ``width``."""

    def __init__(self, order, in_channels, width, bottleneck_ratio=1, group_width=None):
        super().__init__()
        group_width = _resolve_group_width(width, group_width)
        n_groups = width // bottleneck_ratio // group_width
        self.expand = order.conv(in_channels, width, 1)
        self.groups = nn.ModuleList(
            order.conv(width, group_width, 3) for _ in range(n_groups)
        )
        self.project = order.conv(n_groups * group_width, width, 1)
        self.out_channels = width

    def forward(self, x):
        y = F.relu(self.expand(x))
        if len(self.groups) == 1:
            y = F.relu(self.groups[0](y))
        else:
            y = torch.cat([F.relu(group(y)) for group in self.groups], dim=1)
        y = F.relu(self.project(y))
        return y + x


class ResidualStrideBlock(nn.Module):
    """``residual_stride_block`` — downsampling residual block with projected shortcut."""

    def __init__(self, order, in_channels, width, bottleneck_ratio=1, group_width=None, stride=2):
        super().__init__()
        group_width = _resolve_group_width(width, group_width)
        n_groups = width // bottleneck_ratio // group_width
        self.expand = order.conv(in_channels, width, 1)
        self.groups = nn.ModuleList(
            order.conv(width, group_width, 3, stride=stride) for _ in range(n_groups)
        )
        self.project = order.conv(n_groups * group_width, width, 1)
        self.shortcut = order.conv(in_channels, width, 1, stride=stride)
        self.out_channels = width

    def forward(self, x):
        y = F.relu(self.expand(x))
        if len(self.groups) == 1:
            y = F.relu(self.groups[0](y))
        else:
            y = torch.cat([F.relu(group(y)) for group in self.groups], dim=1)
        y = F.relu(self.project(y))
        return y + F.relu(self.shortcut(x))


class ResidualUpsamplingBlock(nn.Module):
    """``residual_upsampling_block`` — upsampling residual block with projected shortcut."""

    def __init__(self, order, in_channels, width, bottleneck_ratio=1, group_width=None, stride=2):
        super().__init__()
        group_width = _resolve_group_width(width, group_width)
        n_groups = width // bottleneck_ratio // group_width
        self.expand = order.conv(in_channels, width, 1)
        self.groups = nn.ModuleList(
            order.deconv(width, group_width, 3, stride) for _ in range(n_groups)
        )
        self.project = order.conv(n_groups * group_width, width, 1)
        self.shortcut = order.deconv(in_channels, width, 1, stride)
        self.out_channels = width

    def forward(self, x):
        y = F.relu(self.expand(x))
        if len(self.groups) == 1:
            y = F.relu(self.groups[0](y))
        else:
            y = torch.cat([F.relu(group(y)) for group in self.groups], dim=1)
        y = F.relu(self.project(y))
        return y + F.relu(self.shortcut(x))


class PSPBranch(nn.Module):
    """One pooling scale of the pyramid pooling module (``interp_block``)."""

    def __init__(self, order, in_channels, pool_size, width, depth,
                 bottleneck_ratio=1, group_width=None):
        super().__init__()
        self.pool_size = pool_size
        self.project = order.conv(in_channels, width, 1) if in_channels != width else None
        self.blocks = nn.ModuleList(
            ResidualBlock(order, width, width, bottleneck_ratio, group_width)
            for _ in range(depth)
        )
        self.out_channels = width

    def forward(self, x):
        if self.project is not None:
            x = self.project(x)
        x = same_avg_pool1d(x, self.pool_size)
        for block in self.blocks:
            x = block(x)
        return upsample_linear(x, self.pool_size)


class PSPModule(nn.Module):
    """``PSPModule`` — concatenation of the input with every pooled scale."""

    def __init__(self, order, in_channels, compression, width, depth):
        super().__init__()
        self.branches = nn.ModuleList(
            PSPBranch(order, in_channels, factor, width, depth) for factor in compression
        )
        self.out_channels = in_channels + width * len(compression)

    def forward(self, x):
        return torch.cat([x] + [branch(x) for branch in self.branches], dim=1)


class Body(nn.Module):
    """``body_uppnet_suppnet`` — the U-shaped backbone with pyramid pooling skips."""

    def __init__(self, order, in_channels, params):
        super().__init__()
        d_i = params["d_i"]
        w_i = params["w_i"]
        psp_bool = params["psp_bool"]
        group_width = params["g"]
        w_ppm = params["w_ppm"]
        d_ppm = params["d_ppm"]
        bottleneck_ratio = 1

        n_stages = (len(d_i) - 1) // 2
        self.n_stages = n_stages
        log2_input_length = 13

        channels = in_channels
        skip_channels = []

        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i in range(n_stages):
            stage = nn.ModuleList()
            for _ in range(d_i[i] - 1):
                stage.append(
                    ResidualBlock(order, channels, w_i[i], bottleneck_ratio, group_width)
                )
                channels = w_i[i]
            self.encoder.append(stage)
            skip_channels.append(channels)
            self.downsample.append(
                ResidualStrideBlock(order, channels, w_i[i + 1], bottleneck_ratio, group_width)
            )
            channels = w_i[i + 1]

        self.middle = nn.ModuleList()
        for _ in range(d_i[n_stages]):
            self.middle.append(
                ResidualBlock(order, channels, w_i[n_stages], bottleneck_ratio, group_width)
            )
            channels = w_i[n_stages]
        skip_channels.append(channels)

        # Pyramid pooling on the skip connections
        self.psp = nn.ModuleList()
        for i in range(n_stages + 1):
            if psp_bool[i] > 0:
                compression = [2 ** (j + 1) for j in range(log2_input_length - i)]
                module = PSPModule(order, skip_channels[i], compression, w_ppm, d_ppm)
                skip_channels[i] = module.out_channels
            else:
                module = nn.Identity()
            self.psp.append(module)

        # Decoder
        channels = skip_channels[n_stages]
        self.upsample = nn.ModuleList()
        self.merge = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.use_skip = []
        for i in range(n_stages):
            width = w_i[i + n_stages + 1]
            self.upsample.append(
                ResidualUpsamplingBlock(order, channels, width, bottleneck_ratio, group_width)
            )
            channels = width
            use_skip = psp_bool[i] > -1
            self.use_skip.append(bool(use_skip))
            if use_skip:
                self.merge.append(
                    order.conv(channels + skip_channels[n_stages - 1 - i], width, 1)
                )
                channels = width
            else:
                self.merge.append(nn.Identity())
            stage = nn.ModuleList()
            for _ in range(d_i[i + n_stages + 1] - 1):
                stage.append(
                    ResidualBlock(order, channels, width, bottleneck_ratio, group_width)
                )
                channels = width
            self.decoder.append(stage)

        self.out_channels = channels

    def forward(self, x):
        skips = []
        for stage, downsample in zip(self.encoder, self.downsample):
            for block in stage:
                x = block(x)
            skips.append(x)
            x = downsample(x)
        for block in self.middle:
            x = block(x)
        skips.append(x)

        skips = [psp(skip) for psp, skip in zip(self.psp, skips)]

        x = skips.pop()
        for i in range(self.n_stages):
            x = self.upsample[i](x)
            if self.use_skip[i]:
                x = torch.cat([x, skips[-1 - i]], dim=1)
                x = F.relu(self.merge[i](x))
            for block in self.decoder[i]:
                x = block(x)
        return x


class Head(nn.Module):
    """``head_continuum`` / ``head_segmentation``."""

    def __init__(self, order, in_channels, name, activation):
        super().__init__()
        self.conv_1 = order.conv(in_channels, 64, 1)
        self.conv_2 = order.conv(64, 32, 1)
        self.conv_3 = order.conv(32, 1, 1, name=name)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = self.conv_3(x)
        return F.relu(x) if self.activation == "relu" else torch.sigmoid(x)


class SUPPNet(nn.Module):
    """Two-stage SUPPNet: a first backbone whose predictions feed a second one.

    Input and output tensors follow the PyTorch convention ``(batch, channels,
    length)``; the Keras model used ``(batch, length, channels)``.
    """

    def __init__(self, params=None, in_channels=1, no_forward_features=9):
        super().__init__()
        params = DEFAULT_PARAMS if params is None else params
        self.in_channels = in_channels
        order = KerasLayerOrder()

        self.body_1 = Body(order, in_channels, params)
        self.cont_1 = Head(order, self.body_1.out_channels, "cont_1", "relu")
        self.seg_1 = Head(order, self.body_1.out_channels, "seg_1", "sigmoid")

        self.forward_features = order.conv(
            self.body_1.out_channels, no_forward_features, 1
        )
        self.body_2 = Body(order, no_forward_features + 2 + in_channels, params)
        self.cont_2 = Head(order, self.body_2.out_channels, "cont_2", "relu")
        self.seg_2 = Head(order, self.body_2.out_channels, "seg_2", "sigmoid")

        self._keras_order = order

    def keras_layer_order(self):
        return self._keras_order

    def load_keras_weights(self, filepath):
        load_keras_weights(self, filepath)
        return self

    def forward(self, x):
        features = self.body_1(x)
        cont_1 = self.cont_1(features)
        seg_1 = self.seg_1(features)

        forwarded = F.relu(self.forward_features(features))
        features = self.body_2(torch.cat([forwarded, cont_1, seg_1, x], dim=1))
        return cont_1, seg_1, self.cont_2(features), self.seg_2(features)


def create_SUPPNet_model(input_shape=(WINDOW_LENGTH, 1)):
    """Build the SUPPNet graph (weights uninitialised).

    ``input_shape`` keeps the Keras ``(length, channels)`` convention; only the
    channel count matters because the network is fully convolutional.
    """
    return SUPPNet(DEFAULT_PARAMS, in_channels=input_shape[-1])


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def select_device(device=None):
    """Pick an inference device: explicit argument, ``SUPPNET_DEVICE``, or the
    best accelerator available (CUDA, Intel XPU, Apple MPS), falling back to CPU.
    """
    if device is None:
        device = os.environ.get("SUPPNET_DEVICE")
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return torch.device("xpu")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_cpu_threads():
    """Intra-op thread count to run SUPPNet with.

    SUPPNet is roughly a thousand very narrow convolutions — most four to
    forty-four channels wide, some only a few dozen samples long — so wall time is
    dominated by per-operation thread synchronisation rather than by arithmetic.
    Measurements bear that out: anything from one thread to about half the cores
    performs within noise of the rest, while spreading every one of those tiny
    operations across *all* cores costs an order of magnitude, and worse on hybrid
    performance/efficiency CPUs where each barrier waits for the slowest core.

    Half the cores, capped at eight, therefore sits comfortably inside the flat
    part of that curve while staying well clear of the cliff.  ``SUPPNET_THREADS``
    overrides it for anyone who wants to tune their own machine.
    """
    return max(1, min((os.cpu_count() or 1) // 2, 8))


def configure_cpu_threads():
    """Apply ``SUPPNET_THREADS``, or fall back to :func:`default_cpu_threads`.

    An explicit ``OMP_NUM_THREADS``/``MKL_NUM_THREADS`` in the environment is taken
    as the user having made the decision already and is left alone.
    """
    requested = os.environ.get("SUPPNET_THREADS", "").strip()
    if requested:
        threads = max(1, int(requested))
    elif any(os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        return torch.get_num_threads()
    else:
        threads = default_cpu_threads()

    torch.set_num_threads(threads)
    return threads


def _default_batch_size(device):
    from_env = os.environ.get("SUPPNET_BATCH_SIZE")
    if from_env:
        return int(from_env)
    return 16 if device.type == "cpu" else 32


class ModelWrapper:
    """Runs SUPPNet over batches of prepared windows.

    ``predict`` accepts and returns NumPy arrays shaped ``(n_windows, length, 1)``
    so that the surrounding pipeline stays framework-agnostic.
    """

    def __init__(self, model, norm_only=True, device=None, batch_size=None):
        self.device = select_device(device)
        self.model = model.to(self.device).eval()
        self.norm_only = norm_only
        self.batch_size = batch_size or _default_batch_size(self.device)
        self.threads = configure_cpu_threads() if self.device.type == "cpu" else None

    @torch.inference_mode()
    def _forward(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[..., None]
        # (n, length, channels) -> (n, channels, length)
        tensor = torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1)))

        continua, segmentations = [], []
        for start in range(0, tensor.shape[0], self.batch_size):
            batch = tensor[start:start + self.batch_size].to(self.device, non_blocking=True)
            _, _, cont, seg = self.model(batch)
            continua.append(cont.float().cpu())
            if not self.norm_only:
                segmentations.append(seg.float().cpu())

        def stack(parts):
            return torch.cat(parts).permute(0, 2, 1).numpy()

        if self.norm_only:
            return stack(continua), None
        return stack(continua), stack(segmentations)

    def predict(self, X):
        continuum, segmentation = self._forward(X)
        if self.norm_only:
            return continuum
        return {"cont": continuum, "seg": segmentation}


# Backwards-compatible alias for the original class name.
modelWrapper = ModelWrapper


def get_suppnet_model(norm_only=True, which_weights="active", device=None, batch_size=None):
    """Build SUPPNet and load one of the bundled weight sets."""
    if which_weights not in WEIGHTS_FILES:
        raise ValueError(
            f"Unknown model type: {which_weights!r}. "
            f"Expected one of {sorted(WEIGHTS_FILES)}."
        )

    script_directory = os.path.dirname(os.path.realpath(__file__))

    if not _env_flag("SUPPNET_ALLOW_TF32"):
        # Keep full float32 precision so results match the TensorFlow reference.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    print("Start creating SUPPNet model!")
    model = create_SUPPNet_model(input_shape=(WINDOW_LENGTH, 1))
    print("SUPPNet model created!")

    print("Start loading weights!")
    model.load_keras_weights(
        os.path.join(script_directory, WEIGHTS_FILES[which_weights])
    )
    print(WEIGHTS_DESCRIPTIONS[which_weights])
    print("Weights loaded!")

    wrapper = ModelWrapper(model, norm_only=norm_only, device=device, batch_size=batch_size)
    threads = f", {wrapper.threads} thread(s)" if wrapper.threads else ""
    print(f"Running on device: {wrapper.device}{threads}, batch size {wrapper.batch_size}")
    return wrapper


if __name__ == "__main__":
    print("Selected device:", select_device())
