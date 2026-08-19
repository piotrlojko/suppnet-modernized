#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the PyTorch SUPPNet port against the saved TensorFlow graph.

The ``.keras`` archives in ``suppnet/supp_models_modernized`` are Keras 2 HDF5
files whose ``model_config`` attribute is a complete description of the original
TensorFlow graph: every layer, its configuration and its inputs.

This script interprets that graph node by node — in Keras' own
``(batch, length, channels)`` layout, with weights matched to layers *by name* —
and compares the result with :class:`suppnet.SUPPNet.SUPPNet`, which builds its
graph from Python code and matches weights *by creation order*.  The two paths
share no code, so agreement between them validates both the architecture and the
weight mapping.

Usage::

    python tools/verify_torch_port.py [--weights active|synth|emission|all]
                                      [--windows 2] [--length 8192] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suppnet.SUPPNet import create_SUPPNet_model  # noqa: E402

MODEL_FILES = {
    "synth": "SUPPNet_synth",
    "active": "SUPPNet_active",
    "emission": "SUPPNet_18_powr",
}
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "suppnet",
    "supp_models_modernized",
)


# --------------------------------------------------------------------------- #
# A literal interpreter for the saved Keras graph
# --------------------------------------------------------------------------- #


def _decode(name):
    return name.decode("utf-8") if isinstance(name, bytes) else name


def read_keras_model(path):
    """Return ``(model_config, {layer_name: [weight arrays]})`` from a .keras file."""
    with h5py.File(path, "r") as handle:
        config = json.loads(handle.attrs["model_config"])
        weights = {}
        group = handle["model_weights"]
        for layer_name in (_decode(n) for n in group.attrs["layer_names"]):
            layer_group = group[layer_name]
            weight_names = [_decode(n) for n in layer_group.attrs["weight_names"]]
            if weight_names:
                weights[layer_name] = [np.array(layer_group[n]) for n in weight_names]
    return config, weights


def _same_pad_total(length, kernel_size, stride):
    out_length = -(-length // stride)
    return max((out_length - 1) * stride + kernel_size - length, 0)


def _activation(x, name):
    if name in (None, "linear"):
        return x
    if name == "relu":
        return F.relu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    raise NotImplementedError(f"activation {name!r}")


def _conv1d(x, config, weights):
    """``keras.layers.Conv1D`` on a ``(batch, length, channels)`` tensor."""
    kernel, bias = weights
    kernel = torch.from_numpy(kernel.transpose(2, 1, 0).copy())
    bias = torch.from_numpy(bias.copy())
    kernel_size = config["kernel_size"][0]
    stride = config["strides"][0]

    x = x.transpose(1, 2)
    if config["padding"] == "same":
        total = _same_pad_total(x.shape[-1], kernel_size, stride)
        if total:
            x = F.pad(x, (total // 2, total - total // 2))
    elif config["padding"] != "valid":
        raise NotImplementedError(config["padding"])
    x = F.conv1d(x, kernel, bias, stride=stride)
    return _activation(x.transpose(1, 2), config.get("activation"))


def _conv1d_transpose(x, config, weights):
    """``keras.layers.Conv1DTranspose`` on a ``(batch, length, channels)`` tensor."""
    kernel, bias = weights
    kernel = torch.from_numpy(kernel.transpose(2, 1, 0).copy())
    bias = torch.from_numpy(bias.copy())
    kernel_size = config["kernel_size"][0]
    stride = config["strides"][0]
    if config["padding"] != "same":
        raise NotImplementedError(config["padding"])

    x = x.transpose(1, 2)
    out_length = x.shape[-1] * stride
    y = F.conv_transpose1d(x, kernel, None, stride=stride)
    # Keras asks tf.nn.conv1d_transpose for an output of input_length * stride;
    # that is the gradient of a SAME-padded forward convolution, whose padding
    # would have been max(kernel_size - stride, 0), smaller half on the left.
    padding = max(kernel_size - stride, 0)
    if padding:
        y = y[..., padding // 2:padding // 2 + out_length]
    elif y.shape[-1] < out_length:
        y = F.pad(y, (0, out_length - y.shape[-1]))
    y = y + bias.view(1, -1, 1)
    return _activation(y.transpose(1, 2), config.get("activation"))


def _average_pooling1d(x, config):
    pool_size = config["pool_size"][0]
    stride = config["strides"][0]
    if config["padding"] != "same":
        raise NotImplementedError(config["padding"])
    x = x.transpose(1, 2)
    length = x.shape[-1]
    total = _same_pad_total(length, pool_size, stride)
    if not total:
        return F.avg_pool1d(x, pool_size, stride).transpose(1, 2)
    left, right = total // 2, total - total // 2
    sums = F.avg_pool1d(F.pad(x, (left, right)), pool_size, stride) * pool_size
    counts = F.avg_pool1d(
        F.pad(torch.ones(1, 1, length), (left, right)), pool_size, stride
    ) * pool_size
    return (sums / counts).transpose(1, 2)


def _up_sampling2d(x, config):
    """``keras.layers.UpSampling2D`` on ``(batch, height, width, channels)``.

    Keras routes bilinear upsampling through ``tf.image.resize``, which uses
    half-pixel centres and no corner alignment.
    """
    if config["interpolation"] != "bilinear":
        raise NotImplementedError(config["interpolation"])
    height_factor, width_factor = config["size"]
    x = x.permute(0, 3, 1, 2)
    size = (x.shape[2] * height_factor, x.shape[3] * width_factor)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


def run_keras_graph(config, weights, x):
    """Execute the saved graph on ``x`` shaped ``(batch, length, channels)``."""
    values = {}

    for layer in config["config"]["layers"]:
        name = layer["name"]
        cls = layer["class_name"]
        layer_config = layer["config"]
        nodes = layer["inbound_nodes"]

        if cls == "InputLayer":
            values[name] = x
            continue

        # Regular layers nest their inputs one level deeper than TFOpLambda ops,
        # which instead carry their call arguments alongside the single input.
        node = nodes[0]
        if cls == "TFOpLambda":
            inputs, call_kwargs = [values[node[0]]], node[3]
        else:
            inputs, call_kwargs = [values[ref[0]] for ref in node], {}

        if cls == "Conv1D":
            values[name] = _conv1d(inputs[0], layer_config, weights[name])
        elif cls == "Conv1DTranspose":
            values[name] = _conv1d_transpose(inputs[0], layer_config, weights[name])
        elif cls == "ReLU":
            assert layer_config["max_value"] is None
            assert layer_config["negative_slope"] == 0.0
            assert layer_config["threshold"] == 0.0
            values[name] = F.relu(inputs[0])
        elif cls == "Add":
            total = inputs[0]
            for other in inputs[1:]:
                total = total + other
            values[name] = total
        elif cls == "Concatenate":
            values[name] = torch.cat(inputs, dim=layer_config["axis"])
        elif cls == "AveragePooling1D":
            values[name] = _average_pooling1d(inputs[0], layer_config)
        elif cls == "UpSampling2D":
            values[name] = _up_sampling2d(inputs[0], layer_config)
        elif cls == "TFOpLambda":
            if layer_config["function"] != "reshape":
                raise NotImplementedError(layer_config["function"])
            values[name] = inputs[0].reshape(call_kwargs["shape"])
        else:
            raise NotImplementedError(cls)

    outputs = [values[ref[0]] for ref in config["config"]["output_layers"]]
    return outputs


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def compare(which, windows, length, seed):
    keras_path = os.path.join(MODEL_DIR, MODEL_FILES[which] + ".keras")
    weights_path = os.path.join(MODEL_DIR, MODEL_FILES[which] + ".weights.h5")

    print(f"\n=== {which} ===")
    print(f"reference graph : {os.path.basename(keras_path)}")
    config, weights = read_keras_model(keras_path)

    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(windows, length, 1, generator=generator)

    with torch.inference_mode():
        reference = run_keras_graph(config, weights, x)

        model = create_SUPPNet_model(input_shape=(length, 1))
        model.load_keras_weights(weights_path)
        model.eval()
        ported = model(x.transpose(1, 2))

    names = ["cont_1", "seg_1", "cont_2", "seg_2"]
    worst = 0.0
    for name, expected, actual in zip(names, reference, ported):
        actual = actual.transpose(1, 2)
        difference = (expected - actual).abs().max().item()
        scale = expected.abs().max().item()
        worst = max(worst, difference)
        print(
            f"  {name:6s} max|Δ| = {difference:.3e}   "
            f"(output range up to {scale:.3f})"
        )
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default="all",
                        choices=sorted(MODEL_FILES) + ["all"])
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    targets = sorted(MODEL_FILES) if args.weights == "all" else [args.weights]
    worst = max(compare(name, args.windows, args.length, args.seed) for name in targets)

    print(f"\nworst deviation across all outputs: {worst:.3e}")
    if worst > args.tolerance:
        print("FAILED: the PyTorch port does not reproduce the TensorFlow graph.")
        return 1
    print("OK: the PyTorch port reproduces the TensorFlow graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
