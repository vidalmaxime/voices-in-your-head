#!/usr/bin/env python3
"""Restore MLX-fused LFM convolution weights to the original HF tensor layout.

Run in the LiquidONNX environment (torch + safetensors). All shapes must match
the original model, apart from MLX's documented convolution axis swap.
"""
import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--fused', type=Path, required=True)
    args = parser.parse_args()
    shapes = {}
    for path in args.base.glob('*.safetensors'):
        with safe_open(path, framework='pt', device='cpu') as f:
            shapes.update({key: tuple(f.get_slice(key).get_shape()) for key in f.keys()})
    shards = list(args.fused.glob('*.safetensors'))
    if not shapes or not shards:
        raise ValueError('Both base and fused safetensors are required')
    found, rewritten = set(), []
    # Validate every shard before modifying anything.
    for path in shards:
        with safe_open(path, framework='pt', device='cpu') as f:
            for key in f.keys():
                shape = tuple(f.get_slice(key).get_shape())
                expected = shapes.get(key)
                if shape != expected:
                    if not ('conv.weight' in key and len(shape) == 3
                            and (shape[0], shape[2], shape[1]) == expected):
                        raise ValueError(f'Unexpected shape: {key}: {shape} vs {expected}')
                found.add(key)
    if found != set(shapes):
        raise ValueError(f'Weight keys differ: {found ^ set(shapes)}')
    for path in shards:
        tensors = {}
        with safe_open(path, framework='pt', device='cpu') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tuple(tensor.shape) != shapes[key]:
                    tensor = tensor.transpose(1, 2).contiguous()
                    rewritten.append(key)
                tensors[key] = tensor.contiguous()
        temporary = path.with_suffix('.safetensors.tmp')
        save_file(tensors, temporary, metadata={'format':'pt'})
        temporary.replace(path)
    print(f'Restored {len(rewritten)} convolution tensors; all {len(found)} weight shapes verified.')


if __name__ == '__main__':
    main()
