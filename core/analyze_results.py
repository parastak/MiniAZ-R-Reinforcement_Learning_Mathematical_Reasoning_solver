import json
import matplotlib.pyplot as plt
import numpy as np
import os
# from collections import defaultdict
from typing import List, Any, Optional
from dataclasses import dataclass

# Constants
LEVEL_MAP = {"easy": 0, "medium": 1, "hard": 2}
LEVEL_LABELS = ["easy", "medium", "hard"]
WINDOW_SIZE = 50


@dataclass
class Task:
    task_id: int
    equation_str: str
    reasoning_type: str
    ground_truth_solution: List[float]
    proposer_notes: Optional[str] = None
    solver_output: Optional[Any] = None
    reward: Optional[float] = None
    meta: Optional[dict] = None


def load_tasks(filepath: str = "logs/reinforce_curriculum2.json") -> List[Task]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training log file not found: {filepath}")

    with open(filepath, "r") as f:
        task_dicts = json.load(f)
    return [Task(**data) for data in task_dicts]


def infer_difficulty(gt_value: float) -> str:
    if abs(gt_value) <= 5:
        return "easy"
    elif abs(gt_value) <= 10:
        return "medium"
    return "hard"


def plot_reward_progress(tasks: List[Task]):
    rewards = [t.reward or 0.0 for t in tasks]

    if len(rewards) < WINDOW_SIZE:
        print(
            f"[plot_reward_progress] Not enough episodes ({len(rewards)}) to compute moving average (window={WINDOW_SIZE})")
        return

    episodes = list(range(len(rewards)))
    moving_avg = np.convolve(rewards, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(episodes[WINDOW_SIZE - 1:], moving_avg, label='Moving Avg Reward')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Progress Over Episodes")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_episode_lengths(tasks: List[Task]):
    episode_lengths = [len(str(t.equation_str)) for t in tasks]  # proxy for complexity
    plt.figure(figsize=(12, 6))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Equation Length")
    plt.title("Episode Length Proxy Over Time")
    plt.grid(True)
    plt.show()


def plot_policy_behavior(tasks: List[Task]):
    preds = [t.solver_output[0] if t.solver_output else 0.0 for t in tasks]
    gts = [t.ground_truth_solution[0] for t in tasks]
    errors = [abs(p - gt) for p, gt in zip(preds, gts)]

    plt.figure(figsize=(12, 6))
    plt.plot(errors, label="Absolute Prediction Error")
    plt.xlabel("Episode")
    plt.ylabel("Error")
    plt.title("Policy Error Progression")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_exploration_vs_exploitation(tasks: List[Task]):
    std_pred = [abs(t.solver_output[0] - t.ground_truth_solution[0]) if t.solver_output else 0.0 for t in tasks]
    plt.figure(figsize=(12, 6))
    plt.plot(std_pred, label='Prediction Std Proxy')
    plt.xlabel("Episode")
    plt.ylabel("Exploration (Error)")
    plt.title("Exploration vs. Exploitation Proxy")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_confidence_interval(tasks: List[Task]):
    rewards = [t.reward or 0.0 for t in tasks]
    episodes = np.arange(len(rewards))
    window = WINDOW_SIZE

    if len(rewards) < window:
        print(f"[plot_confidence_interval] Not enough episodes ({len(rewards)}) to compute moving CI (window={window})")
        return

    means = []
    cis = []
    for i in range(len(rewards) - window):
        windowed = rewards[i:i + window]
        mean = np.mean(windowed)
        ci = 1.96 * np.std(windowed) / np.sqrt(window)
        means.append(mean)
        cis.append(ci)

    episodes = episodes[window:]
    means = np.array(means)
    cis = np.array(cis)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, means, label="Mean Reward")
    plt.fill_between(episodes, means - cis, means + cis, color='b', alpha=0.2, label="95% CI")
    plt.title("Reward Confidence Interval Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_entropy_and_logstd(tasks: List[Task]):
    log_stds = [t.meta.get("log_std") if t.meta and "log_std" in t.meta else np.nan for t in tasks]
    entropies = [t.meta.get("entropy") if t.meta and "entropy" in t.meta else np.nan for t in tasks]

    episodes = np.arange(len(tasks))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Log Std', color=color)
    ax1.plot(episodes, log_stds, label="log_std", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([-5, 2])

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Entropy', color=color)
    ax2.plot(episodes, entropies, label="entropy", color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Exploration Signals: log_std and Entropy Over Time")
    fig.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        tasks = load_tasks()
        print(f"[INFO] Loaded {len(tasks)} tasks.")

        if len(tasks) < WINDOW_SIZE:
            print(f"[WARNING] Only {len(tasks)} tasks found. Plots using window={WINDOW_SIZE} may not render.")

        plot_reward_progress(tasks)
        plot_episode_lengths(tasks)
        plot_policy_behavior(tasks)
        plot_exploration_vs_exploitation(tasks)
        plot_confidence_interval(tasks)
        plot_entropy_and_logstd(tasks)

    except Exception as e:
        print(f"Error: {e}")
