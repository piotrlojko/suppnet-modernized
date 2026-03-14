### The data located in this folder contains modern TF-friendly model checkpoints, which have been converted from (/supp_weights):
* SUPPNet_18_powr.data-00000-of-00001
* SUPPNet_18_powr.index
* SUPPNet_active.data-00000-of-00001
* SUPPNet_active.index
* SUPPNet_synth.data-00000-of-00001
* SUPPNet_synth.index
  
### into (/supp_models_modernized):
* SUPPNet_synth.weights.h5
* SUPPNet_synth.metadata.json
* SUPPNet_synth.keras
* SUPPNet_active.weights.h5
* SUPPNet_active.metadata.json
* SUPPNet_active.keras
* SUPPNet_18_powr.weights.h5
* SUPPNet_18_powr.metadata.json
* SUPPNet_18_powr.keras

### using the following code:
```
#!/usr/bin/env python3
"""
Convert legacy TF checkpoint-prefix weights from SUPPNet into modern formats:
- Keras v3 archive: .keras
- Keras HDF5 weights: .weights.h5

Run this script in the legacy environment (Python 3.8 + TF 2.4) so old checkpoint
loading works exactly as in the original repo.
"""

import os
import json
import argparse
import tensorflow as tf

from suppnet.SUPPNet import create_SUPPNet_model

LEGACY_PREFIXES = [
    "SUPPNet_18_powr",
    "SUPPNet_active",
    "SUPPNet_synth",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_model():
    # Must match original architecture exactly
    model = create_SUPPNet_model(input_shape=(8192, 1))
    # build variables
    _ = model(tf.zeros((1, 8192, 1), dtype=tf.float32), training=False)
    return model


def convert_one(weights_dir: str, out_dir: str, prefix_name: str):
    ckpt_prefix = os.path.join(weights_dir, prefix_name)
    index_file = ckpt_prefix + ".index"
    data_file = ckpt_prefix + ".data-00000-of-00001"

    if not (os.path.exists(index_file) and os.path.exists(data_file)):
        raise FileNotFoundError(
            f"Missing checkpoint pair for {prefix_name}:\n"
            f"  {index_file}\n"
            f"  {data_file}"
        )

    print(f"\n=== Converting {prefix_name} ===")
    print(f"Loading legacy checkpoint prefix: {ckpt_prefix}")

    model = build_model()
    status = model.load_weights(ckpt_prefix)
    # In TF 2.4 this is Trackable status. expect_partial() is safe here.
    try:
        status.expect_partial()
    except Exception:
        pass

    # Export modern-friendly files
    keras_path = os.path.join(out_dir, f"{prefix_name}.keras")
    h5_weights_path = os.path.join(out_dir, f"{prefix_name}.weights.h5")
    metadata_path = os.path.join(out_dir, f"{prefix_name}.metadata.json")

    print(f"Saving .keras archive: {keras_path}")
    model.save(keras_path)  # full model config + weights

    print(f"Saving .weights.h5: {h5_weights_path}")
    model.save_weights(h5_weights_path)

    # Save minimal metadata for reproducibility
    metadata = {
        "source_checkpoint_prefix": ckpt_prefix,
        "source_files": [index_file, data_file],
        "exported_files": [keras_path, h5_weights_path],
        "model_input_shape": [8192, 1],
        "model_outputs": ["cont_1", "seg_1", "cont_2", "seg_2"],
        "tensorflow_version_used_for_export": tf.__version__,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote metadata: {metadata_path}")
    print(f"Done: {prefix_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert legacy SUPPNet checkpoint-prefix weights to modern Keras formats."
    )
    parser.add_argument(
        "--weights-dir",
        default=os.path.join("suppnet", "supp_weights"),
        help="Directory containing legacy checkpoint files (.index + .data-00000-of-00001).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("suppnet", "supp_weights_converted"),
        help="Directory for converted files.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of prefixes to convert, e.g. --only SUPPNet_active",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    targets = args.only if args.only else LEGACY_PREFIXES

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Input dir:  {os.path.abspath(args.weights_dir)}")
    print(f"Output dir: {os.path.abspath(args.out_dir)}")
    print(f"Targets: {targets}")

    for prefix in targets:
        convert_one(args.weights_dir, args.out_dir, prefix)

    print("\nAll requested conversions finished.")


if __name__ == "__main__":
    main()
```
