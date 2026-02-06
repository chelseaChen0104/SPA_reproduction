#!/usr/bin/env python3
"""
Download SPA Pre-trained Data from HuggingFace

Usage:
    python scripts/download_data.py --env sokoban
    python scripts/download_data.py --env all --download-models
"""

import argparse
import os

DATASETS = {
    'sokoban': 'tyzhu/SPA-sokoban-data',
    'frozenlake': 'tyzhu/SPA-frozenlake-data',
    'sudoku': 'tyzhu/SPA-sudoku-data',
}

MODELS = {
    'sokoban': 'tyzhu/SPA-sokoban-qwen2.5-1.5b-instruct',
    'frozenlake': 'tyzhu/SPA-frozenlake-qwen2.5-1.5b-instruct',
    'sudoku': 'tyzhu/SPA-sudoku-qwen2.5-1.5b-instruct',
}


def download(repo_id: str, output_dir: str, repo_type: str = 'dataset'):
    from huggingface_hub import snapshot_download
    local_dir = os.path.join(output_dir, repo_id.split('/')[-1])
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading {repo_id}...")
    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir)
    print(f"Saved to {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='sokoban',
                        choices=['sokoban', 'frozenlake', 'sudoku', 'all'])
    parser.add_argument('--output-dir', type=str, default='data')
    parser.add_argument('--download-models', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    envs = list(DATASETS.keys()) if args.env == 'all' else [args.env]

    for env in envs:
        download(DATASETS[env], args.output_dir, 'dataset')
        if args.download_models:
            download(MODELS[env], args.output_dir, 'model')

    print("\nDone!")


if __name__ == '__main__':
    main()
