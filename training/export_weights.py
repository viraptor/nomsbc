"""
Export trained PyTorch weights to a flat binary format for the C runtime.

The C inference code reads weights as contiguous float32 arrays in a
specific order matching the model architecture. This script handles
the conversion.

Output format: flat binary file of float32 values in this order:
  1. FeatureEncoder weights (linear1.W, linear1.b, linear2.W, linear2.b)
  2. GRU weights (per layer: W_ih, W_hh, b_ih, b_hh)
  3. ParameterHead weights (linear.W, linear.b)
"""

import argparse
import struct
import numpy as np
import torch
from model import NoLACEmSBC


def export_model(checkpoint_path, output_path):
    model = NoLACEmSBC()
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # Handle both raw state_dict and wrapped checkpoint formats
    if 'generator' in state:
        state = state['generator']
    model.load_state_dict(state)

    weights = []
    param_names = []

    # Write in architecture order
    for name, param in model.named_parameters():
        data = param.detach().cpu().numpy().flatten()
        weights.append(data)
        param_names.append((name, data.shape[0]))

    all_weights = np.concatenate(weights).astype(np.float32)

    with open(output_path, 'wb') as f:
        # Header: magic + version + total weight count
        f.write(struct.pack('<4sII', b'NMSB', 1, len(all_weights)))
        # Layer table: num_layers, then (name_len, name, count) per layer
        f.write(struct.pack('<I', len(param_names)))
        for name, count in param_names:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', count))
        # Weights
        f.write(all_weights.tobytes())

    total_params = sum(c for _, c in param_names)
    print(f"Exported {total_params:,} parameters ({len(all_weights) * 4:,} bytes)")
    print(f"Layers:")
    for name, count in param_names:
        print(f"  {name}: {count:,}")
    print(f"\nWritten to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export model weights for C runtime')
    parser.add_argument('checkpoint', help='PyTorch checkpoint path')
    parser.add_argument('-o', '--output', default='nomsbc_weights.bin',
                        help='Output binary file')
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
