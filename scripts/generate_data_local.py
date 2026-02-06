#!/usr/bin/env python3
"""
Local SPA-format Data Generation (No GPU/LLM required)

Generates trajectories using BFS solver or random policy,
formatted in SPA's world model format.

Usage:
    python scripts/generate_data_local.py --env sokoban --num-trajectories 1000
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
from sklearn.model_selection import train_test_split


# ============================================================================
# Sokoban Environment (Minimal Implementation)
# ============================================================================

class SokobanEnv:
    """Minimal Sokoban environment for data generation."""

    GRID_LOOKUP = {0: '#', 1: '_', 2: 'O', 3: 'V', 4: 'X', 5: 'P', 6: '@'}
    ACTION_LOOKUP = {1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'}
    ACTION_DELTA = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def __init__(self, dim=(6, 6), num_boxes=1, max_steps=100):
        self.dim = dim
        self.num_boxes = num_boxes
        self.max_steps = max_steps
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Simple room generation
        self.room = np.ones(self.dim, dtype=np.int32)
        self.room[0, :] = 0  # walls
        self.room[-1, :] = 0
        self.room[:, 0] = 0
        self.room[:, -1] = 0

        # Place goal, box, player
        empty = [(i, j) for i in range(1, self.dim[0]-1)
                 for j in range(1, self.dim[1]-1)]
        random.shuffle(empty)

        self.goals = [empty.pop()]
        self.room[self.goals[0]] = 2

        self.boxes = [empty.pop()]
        self.player = empty.pop()

        self.steps = 0
        self.done = False
        return self.render()

    def render(self) -> str:
        grid = self.room.copy()
        for box in self.boxes:
            if grid[box] == 2:
                grid[box] = 3  # box on goal
            else:
                grid[box] = 4
        if grid[self.player] == 2:
            grid[self.player] = 6  # player on goal
        else:
            grid[self.player] = 5

        lines = [''.join(self.GRID_LOOKUP[c] for c in row) for row in grid]
        return '\n'.join(lines)

    def render_with_coords(self) -> str:
        grid_str = self.render()
        coords = [f"Player (P) is at {self.player}"]
        for i, box in enumerate(self.boxes):
            coords.append(f"box (X) is at {box}")
        for i, goal in enumerate(self.goals):
            coords.append(f"target (O) is at {goal}")
        return grid_str + "\n" + "; ".join(coords) + "."

    def step(self, action: int) -> Tuple[str, float, bool, Dict]:
        if self.done:
            return self.render(), 0, True, {}

        self.steps += 1
        dx, dy = self.ACTION_DELTA[action]
        new_pos = (self.player[0] + dx, self.player[1] + dy)

        # Check wall
        if self.room[new_pos] == 0:
            return self.render(), 0, False, {'valid': False}

        # Check box push
        if new_pos in self.boxes:
            box_new = (new_pos[0] + dx, new_pos[1] + dy)
            if self.room[box_new] == 0 or box_new in self.boxes:
                return self.render(), 0, False, {'valid': False}
            self.boxes.remove(new_pos)
            self.boxes.append(box_new)

        self.player = new_pos

        # Check win
        success = all(box in self.goals for box in self.boxes)
        if success:
            self.done = True
            return self.render(), 1.0, True, {'success': True}

        if self.steps >= self.max_steps:
            self.done = True
            return self.render(), 0, True, {'success': False}

        return self.render(), 0, False, {'valid': True}

    def get_valid_actions(self) -> List[int]:
        valid = []
        for action in [1, 2, 3, 4]:
            dx, dy = self.ACTION_DELTA[action]
            new_pos = (self.player[0] + dx, self.player[1] + dy)
            if self.room[new_pos] != 0:
                valid.append(action)
        return valid


def bfs_solve(env) -> Optional[List[int]]:
    """BFS solver to find optimal action sequence."""
    initial_state = (env.player, tuple(sorted(env.boxes)))
    queue = deque([(initial_state, [])])
    visited = {initial_state}

    while queue:
        (player, boxes), actions = queue.popleft()

        if len(actions) > 50:  # depth limit
            continue

        for action in [1, 2, 3, 4]:
            dx, dy = env.ACTION_DELTA[action]
            new_player = (player[0] + dx, player[1] + dy)

            if env.room[new_player] == 0:
                continue

            new_boxes = list(boxes)
            if new_player in boxes:
                box_new = (new_player[0] + dx, new_player[1] + dy)
                if env.room[box_new] == 0 or box_new in boxes:
                    continue
                new_boxes.remove(new_player)
                new_boxes.append(box_new)

            new_boxes = tuple(sorted(new_boxes))
            new_state = (new_player, new_boxes)

            if new_state in visited:
                continue
            visited.add(new_state)

            new_actions = actions + [action]

            if all(box in env.goals for box in new_boxes):
                return new_actions

            queue.append((new_state, new_actions))

    return None


# ============================================================================
# SPA Format Data Generation
# ============================================================================

SYSTEM_PROMPT = """You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets. You are provided with a symbol grid and the zero-indexed coordinates of the player, each box, and each target.

Symbol meanings: #=wall, _=empty, O=target, X=box, P=player, V=box on target

In your reasoning:
1. Describe the current state in <observation>
2. Predict the next state in <prediction>

Then provide your action in <answer>."""


def format_spa_response(observation: str, prediction: str, action: str) -> str:
    """Format a single turn in SPA format."""
    return f"""<think> <observation>{observation}</observation> <prediction>{prediction}</prediction> </think>
<answer> {action} </answer>"""


def generate_trajectory(env, policy='bfs', seed=None) -> Optional[Dict]:
    """Generate a single trajectory with SPA format."""
    state = env.reset(seed=seed)

    if policy == 'bfs':
        actions = bfs_solve(env)
        if actions is None:
            return None
    else:
        actions = None  # will use random

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    real_states = [env.render_with_coords()]

    step = 0
    done = False

    while not done:
        # Get action
        if actions is not None:
            if step >= len(actions):
                break
            action = actions[step]
        else:
            valid = env.get_valid_actions()
            action = random.choice(valid) if valid else 1

        action_name = env.ACTION_LOOKUP[action]
        current_obs = env.render_with_coords()

        # Take step
        next_state, reward, done, info = env.step(action)
        next_obs = env.render_with_coords()

        # Add user turn (current state)
        messages.append({
            "role": "user",
            "content": f"Current state:\n{current_obs}"
        })

        # Add assistant turn (observation + prediction + action)
        response = format_spa_response(current_obs, next_obs, action_name)
        messages.append({
            "role": "assistant",
            "content": response
        })

        real_states.append(next_obs)
        step += 1

        if step > 50:
            break

    success = info.get('success', False) if info else False

    return {
        'messages_list': messages,
        'real_states': real_states,
        'success': success,
        'num_steps': step
    }


def convert_to_sft_format(trajectories: List[Dict]) -> pd.DataFrame:
    """Convert trajectories to SFT DataFrame format."""
    rows = []

    for traj in trajectories:
        messages = traj['messages_list']

        for i, msg in enumerate(messages):
            if msg['role'] == 'assistant':
                prompt = np.array(messages[:i])
                rows.append({
                    'data_source': 'sokoban',
                    'prompt': prompt,
                    'response': msg['content'],
                    'ability': 'world_model',
                    'reward_model': "{'style': 'rule'}",
                    'extra_info': json.dumps({'success': traj['success']})
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Generate SPA-format data locally')
    parser.add_argument('--env', type=str, default='sokoban', choices=['sokoban'])
    parser.add_argument('--num-trajectories', type=int, default=500)
    parser.add_argument('--policy', type=str, default='bfs', choices=['bfs', 'random'])
    parser.add_argument('--output-dir', type=str, default='data/local_generated')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-split', type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 60)
    print("Local SPA Data Generation")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Trajectories: {args.num_trajectories}")
    print(f"Policy: {args.policy}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = SokobanEnv(dim=(6, 6), num_boxes=1)
    trajectories = []

    print("\nGenerating trajectories...")
    for i in range(args.num_trajectories):
        traj = generate_trajectory(env, policy=args.policy, seed=args.seed + i)
        if traj is not None:
            trajectories.append(traj)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{args.num_trajectories}")

    print(f"\nGenerated {len(trajectories)} valid trajectories")

    # Save raw trajectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(args.output_dir, f'raw_trajectories_{timestamp}.json')
    with open(raw_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_trajs = []
        for t in trajectories:
            json_trajs.append({
                'messages_list': t['messages_list'],
                'success': t['success'],
                'num_steps': t['num_steps']
            })
        json.dump(json_trajs, f, indent=2)
    print(f"Saved raw trajectories to {raw_path}")

    # Convert to SFT format
    print("\nConverting to SFT format...")
    df = convert_to_sft_format(trajectories)

    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=42)

    train_df.to_parquet(os.path.join(args.output_dir, 'wm_train.parquet'))
    val_df.to_parquet(os.path.join(args.output_dir, 'wm_val.parquet'))
    train_df.to_csv(os.path.join(args.output_dir, 'wm_train.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'wm_val.csv'), index=False)

    # Statistics
    success_count = sum(1 for t in trajectories if t['success'])
    avg_steps = sum(t['num_steps'] for t in trajectories) / len(trajectories)

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Trajectories: {len(trajectories)}")
    print(f"Success rate: {100*success_count/len(trajectories):.1f}%")
    print(f"Avg steps: {avg_steps:.1f}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"\nOutput: {args.output_dir}")


if __name__ == '__main__':
    main()
