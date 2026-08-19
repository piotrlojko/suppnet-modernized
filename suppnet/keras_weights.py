#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Loading of the original Keras/TensorFlow SUPPNet weight files.

The checkpoints shipped with this repository are Keras 2 HDF5 weight files.  They
store one group per layer, named with Keras' automatic naming scheme
(``conv1d_1033``, ``conv1d_transpose_32``, ...) plus the four explicitly named
output layers (``cont_1``, ``seg_1``, ``cont_2``, ``seg_2``).

Keras derives those automatic names from a per-class counter incremented on every
instantiation, so sorting them numerically recovers the order in which the layers
were created.  The PyTorch port records the same creation order while building
the graph (see :class:`suppnet.SUPPNet.KerasLayerOrder`), which is what makes a
rename-free, one-to-one mapping possible.

Kernel layouts differ between the two frameworks:

===================  ==========================  ==========================
layer                Keras kernel                PyTorch weight
===================  ==========================  ==========================
``Conv1D``           ``(k, in_ch, out_ch)``      ``(out_ch, in_ch, k)``
``Conv1DTranspose``  ``(k, out_ch, in_ch)``      ``(in_ch, out_ch, k)``
===================  ==========================  ==========================

Both are recovered by transposing the axes to ``(2, 1, 0)``.
"""

from __future__ import annotations

import os
import re

import h5py
import numpy as np
import torch

__all__ = ["read_keras_weight_file", "load_keras_weights"]


_AUTO_CONV = re.compile(r"^conv1d(?:_(\d+))?$")
_AUTO_DECONV = re.compile(r"^conv1d_transpose(?:_(\d+))?$")


def _decode(name):
    return name.decode("utf-8") if isinstance(name, bytes) else name


def read_keras_weight_file(filepath):
    """Read a Keras 2 HDF5 weight file into ``{layer_name: [arrays]}``."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Weights file not found: {filepath}")

    layers = {}
    with h5py.File(filepath, "r") as handle:
        for layer_name in (_decode(n) for n in handle.attrs["layer_names"]):
            group = handle[layer_name]
            weight_names = [_decode(n) for n in group.attrs["weight_names"]]
            if weight_names:
                layers[layer_name] = [np.array(group[n]) for n in weight_names]
    return layers


def _sorted_by_index(names, pattern):
    matches = [(name, pattern.match(name)) for name in names]
    return [
        name
        for name, match in sorted(
            ((name, match) for name, match in matches if match),
            key=lambda item: int(item[1].group(1) or 0),
        )
    ]


def _assign(parameter, array, layer_name, what):
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tuple(parameter.shape) != tuple(tensor.shape):
        raise ValueError(
            f"Shape mismatch for {what} of Keras layer {layer_name!r}: "
            f"checkpoint has {tuple(tensor.shape)}, model expects {tuple(parameter.shape)}."
        )
    with torch.no_grad():
        parameter.copy_(tensor)


def _load_layer(module, arrays, layer_name):
    if len(arrays) != 2:
        raise ValueError(
            f"Keras layer {layer_name!r} has {len(arrays)} weight tensors, expected 2 "
            "(kernel and bias)."
        )
    kernel, bias = arrays
    _assign(module.weight, kernel.transpose(2, 1, 0), layer_name, "kernel")
    bias_parameter = getattr(module, "tf_bias", None)
    if bias_parameter is None:
        bias_parameter = module.bias
    _assign(bias_parameter, bias, layer_name, "bias")


def load_keras_weights(model, filepath):
    """Copy every weight of a Keras SUPPNet checkpoint into its PyTorch counterpart.

    Raises if the checkpoint and the model disagree on the number of layers or on
    any tensor shape, so a silently mismatched load is not possible.
    """
    layers = read_keras_weight_file(filepath)
    order = model.keras_layer_order()

    conv_names = _sorted_by_index(layers, _AUTO_CONV)
    deconv_names = _sorted_by_index(layers, _AUTO_DECONV)
    named = sorted(order.named_layers)

    expected = len(order.conv_layers) + len(order.deconv_layers) + len(named)
    if len(layers) != expected:
        raise ValueError(
            f"{os.path.basename(filepath)} holds {len(layers)} weighted layers, "
            f"but the model has {expected}."
        )
    for kind, found, wanted in (
        ("Conv1D", conv_names, order.conv_layers),
        ("Conv1DTranspose", deconv_names, order.deconv_layers),
    ):
        if len(found) != len(wanted):
            raise ValueError(
                f"{os.path.basename(filepath)} holds {len(found)} {kind} layers, "
                f"but the model has {len(wanted)}."
            )

    for module, layer_name in zip(order.conv_layers, conv_names):
        _load_layer(module, layers[layer_name], layer_name)
    for module, layer_name in zip(order.deconv_layers, deconv_names):
        _load_layer(module, layers[layer_name], layer_name)
    for layer_name in named:
        if layer_name not in layers:
            raise ValueError(
                f"{os.path.basename(filepath)} is missing the {layer_name!r} layer."
            )
        _load_layer(order.named_layers[layer_name], layers[layer_name], layer_name)

    return model
