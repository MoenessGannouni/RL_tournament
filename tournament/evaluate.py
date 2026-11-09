#!/usr/bin/env python3
"""
RL Course Tournament Evaluator
Author: Course Teaching Staff
Description: Evaluates RL agent submissions for Lab 1 (LunarLander-v3)
and Lab 2 (Taxi-v3/v4), generating score reports and rankings.
"""

from functools import wraps

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Optional gymnasium import (required only during active agent simulation)
try:
    import gymnasium as gym
except ImportError:
    gym = None


LAB_CONFIGS = {
    1: {
        "name": "Lab 1 - LunarLander-v3",
        "env_candidates": ["LunarLander-v3", "LunarLander-v2"],
        "max_steps": 1000,
        "default_episodes": 30,
        "pass_threshold": 100.0,
        "mastery_threshold": 200.0,
        "is_discrete_obs": False,
        "expected_action_space": 4,
    },
    2: {
        "name": "Lab 2 - Taxi-v3 / Taxi-v4",
        "env_candidates": ["Taxi-v4", "Taxi-v3"],
        "max_steps": 200,
        "default_episodes": 50,
        "pass_threshold": 4.5,
        "mastery_threshold": 7.0,
        "is_discrete_obs": True,
        "expected_action_space": 6,
    },
}


# Backward-compatible name for existing notebooks and integrations.
PRACTICAL_CONFIGS = LAB_CONFIGS


def _accept_legacy_practical_id(lab_position=0):
    """Keep the old keyword working while exposing lab_id in Python signatures."""
    def decorate(function):
        @wraps(function)
        def compatible(*args, **kwargs):
            if "practical_id" in kwargs:
                if "lab_id" in kwargs or len(args) > lab_position:
                    raise TypeError("Specify lab_id or practical_id, not both")
                kwargs["lab_id"] = kwargs.pop("practical_id")
            return function(*args, **kwargs)
        return compatible
    return decorate


def make_env(env_candidates, seed=None):
    """Attempt to create the environment using candidate names."""
    last_err = None
    for env_id in env_candidates:
        try:
            env = gym.make(env_id)
            if seed is not None:
                env.reset(seed=seed)
            return env, env_id
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not instantiate any of {env_candidates}: {last_err}")


def load_agent_function(submission_dir):
    """Dynamically loads the agent function from agent.py in submission_dir."""
    submission_path = Path(submission_dir).resolve()
    agent_file = submission_path / "agent.py"

    if not agent_file.exists():
        raise FileNotFoundError(f"Missing required file: {agent_file}")

    # Temporarily set cwd and sys.path to submission dir so relative paths work
    orig_cwd = os.getcwd()
    sys_path_inserted = False
    sub_dir_str = str(submission_path)

    if sub_dir_str not in sys.path:
        sys.path.insert(0, sub_dir_str)
        sys_path_inserted = True

    try:
        os.chdir(sub_dir_str)
        try:
            from stable_baselines3 import PPO
            _orig_ppo_load = PPO.load
            def _safe_ppo_load(path, *args, **kwargs):
                custom = kwargs.setdefault('custom_objects', {})
                custom.setdefault('learning_rate', 0.0)
                custom.setdefault('lr_schedule', lambda _: 0.0)
                custom.setdefault('clip_range', lambda _: 0.2)
                return _orig_ppo_load(path, *args, **kwargs)
            PPO.load = _safe_ppo_load
        except ImportError:
            pass
        spec = importlib.util.spec_from_file_location("student_agent_module", str(agent_file))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load specification for {agent_file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "agent"):
            raise AttributeError("agent.py must define a function named 'agent(observation)'")

        agent_fn = getattr(module, "agent")
        if not callable(agent_fn):
            raise TypeError("'agent' attribute in agent.py must be callable")

        return agent_fn
    finally:
        os.chdir(orig_cwd)
        if sys_path_inserted and sub_dir_str in sys.path:
            sys.path.remove(sub_dir_str)


@_accept_legacy_practical_id(lab_position=1)
def evaluate_agent(agent_fn, lab_id, num_episodes=30, base_seed=42):
    """
    Evaluates an agent over a fixed set of seeded episodes.
    Returns detailed metrics dictionary.
    """
    config = LAB_CONFIGS[lab_id]
    max_steps = config["max_steps"]

    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    step_times = []

    active_env_id = None

    for ep in range(num_episodes):
        ep_seed = base_seed + ep
        env, active_env_id = make_env(config["env_candidates"])
        obs, info = env.reset(seed=ep_seed)

        ep_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < max_steps:
            t0 = time.perf_counter()
            try:
                action = agent_fn(obs)
            except Exception as e:
                env.close()
                raise RuntimeError(f"Error in agent(observation) call at episode {ep}, step {steps}: {e}")
            t1 = time.perf_counter()
            step_times.append(t1 - t0)

            # Cast action to standard int
            try:
                action = int(action)
            except Exception:
                env.close()
                raise TypeError(f"agent() returned non-integer action: {action} (type: {type(action)})")

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1

        env.close()
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)

        # Success criteria
        if lab_id == 1:
            # LunarLander: score >= 200 indicates safe landing
            episode_successes.append(1 if ep_reward >= 200.0 else 0)
        else:
            # Taxi: positive score means successfully delivered passenger
            episode_successes.append(1 if ep_reward > 0.0 else 0)

    rewards_arr = np.array(episode_rewards)
    lengths_arr = np.array(episode_lengths)

    mean_reward = float(np.mean(rewards_arr))
    std_reward = float(np.std(rewards_arr))
    min_reward = float(np.min(rewards_arr))
    max_reward = float(np.max(rewards_arr))
    median_reward = float(np.median(rewards_arr))
    success_rate = float(np.mean(episode_successes) * 100.0)
    avg_steps = float(np.mean(lengths_arr))
    avg_step_ms = float(np.mean(step_times) * 1000.0) if step_times else 0.0

    # Determine status
    if mean_reward >= config["mastery_threshold"]:
        status = "MASTERY 🏆"
        status_code = "passed"
    elif mean_reward >= config["pass_threshold"]:
        status = "PASSED ✅"
        status_code = "passed"
    else:
        status = "NEEDS IMPROVEMENT ⚠️"
        status_code = "warning"

    return {
        "lab_id": lab_id,
        "lab_name": config["name"],
        "env_id": active_env_id,
        "num_episodes": num_episodes,
        "base_seed": base_seed,
        "mean_reward": round(mean_reward, 2),
        "std_reward": round(std_reward, 2),
        "min_reward": round(min_reward, 2),
        "max_reward": round(max_reward, 2),
        "median_reward": round(median_reward, 2),
        "success_rate": round(success_rate, 1),
        "avg_steps": round(avg_steps, 1),
        "avg_step_ms": round(avg_step_ms, 3),
        "status": status,
        "status_code": status_code,
        "pass_threshold": config["pass_threshold"],
        "mastery_threshold": config["mastery_threshold"],
    }


@_accept_legacy_practical_id(lab_position=1)
def evaluate_single_submission(submission_path, lab_id, num_episodes=30, base_seed=42,
                               isolated=False, sandbox_image="rl-tournament-agent:local"):
    """Evaluates a single submission directory and returns report dict."""
    sub_path = Path(submission_path).resolve()
    participant_name = sub_path.name

    result = {
        "participant": participant_name,
        "submission_path": str(sub_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "success": False,
        "error": None,
    }

    agent_fn = None
    try:
        if isolated:
            from sandbox import IsolatedAgent
            agent_fn = IsolatedAgent(submission_path, LAB_CONFIGS[lab_id]["expected_action_space"],
                                     image=sandbox_image)
        else:
            agent_fn = load_agent_function(sub_path)
        metrics = evaluate_agent(agent_fn, lab_id, num_episodes, base_seed)
        result.update(metrics)
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "ERROR ❌"
        result["status_code"] = "error"
        result["mean_reward"] = -9999.0
        result["std_reward"] = 0.0
    finally:
        if isolated and agent_fn is not None:
            agent_fn.close()

    return result


def format_markdown_report(result):
    """Formats a single evaluation result as clean GitHub-ready Markdown."""
    if not result["success"]:
        return f"""### ❌ Evaluation Failed for `{result['participant']}`
**Error:** `{result['error']}`

Please verify that:
1. `agent.py` exists and defines `def agent(observation): ...`
2. All necessary model weights (e.g. `.zip` or `.pkl`) are committed in your folder.
3. Your agent does not exceed time/memory limits.
"""

    status_icon = "🏆" if "MASTERY" in result["status"] else ("✅" if "PASSED" in result["status"] else "⚠️")

    return f"""### {status_icon} RL Tournament Evaluation Report: `{result['participant']}`

**Assignment:** {result.get('lab_name', result.get('practical_name', 'Unknown lab'))} (`{result['env_id']}`)<br>
**Status:** **{result['status']}** (Threshold: $\\ge {result['pass_threshold']}$ pass, $\\ge {result['mastery_threshold']}$ mastery)  
**Evaluated Episodes:** {result['num_episodes']} (Base Seed: {result['base_seed']})

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Mean Reward** | **`{result['mean_reward']:.2f}`** | Official Ranking Metric |
| **Std Deviation** | `±{result['std_reward']:.2f}` | Stability / Variance |
| **Median Reward** | `{result['median_reward']:.2f}` | Outlier-robust midpoint |
| **Min / Max Reward** | `[{result['min_reward']:.2f}, {result['max_reward']:.2f}]` | Worst vs Best Run |
| **Success Rate** | **`{result['success_rate']:.1f}%`** | Objective completed |
| **Avg Steps / Episode** | `{result['avg_steps']:.1f}` | Efficiency |
| **Inference Latency** | `{result['avg_step_ms']:.2f} ms` | Step compute speed |

*Evaluated automatically by GitHub Actions Tournament Runner at {result['timestamp']}.*
"""


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Course Submissions")
    parser.add_argument("--lab", "--practical", dest="lab", type=int, choices=[1, 2], required=True, help="Lab ID (1 or 2)")
    parser.add_argument("--submission", type=str, help="Path to single submission folder")
    parser.add_argument("--all", action="store_true", help="Evaluate all submissions in lab folder")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for evaluation")
    parser.add_argument("--output-json", type=str, help="Path to write JSON results file")
    parser.add_argument("--output-pr-comment", type=str, help="Path to write markdown comment for PR")
    parser.add_argument("--isolated", action="store_true", help="Require Docker-isolated agent inference (fails closed)")
    parser.add_argument("--sandbox-image", default="rl-tournament-agent:local")
    args = parser.parse_args()
    if args.episodes is not None and not 1 <= args.episodes <= 100:
        parser.error("episodes must be between 1 and 100")

    lab_id = args.lab
    episodes = args.episodes or LAB_CONFIGS[lab_id]["default_episodes"]

    if args.submission:
        res = evaluate_single_submission(args.submission, lab_id, episodes, args.seed,
                                         isolated=args.isolated, sandbox_image=args.sandbox_image)
        print(format_markdown_report(res))

        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

        if args.output_pr_comment:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_pr_comment)), exist_ok=True)
            with open(args.output_pr_comment, "w", encoding="utf-8") as f:
                f.write(format_markdown_report(res))

        if not res["success"]:
            sys.exit(1)

    elif args.all:
        submissions_dir = Path("submissions") / f"lab{lab_id}"
        if not submissions_dir.exists():
            print(f"Directory {submissions_dir} does not exist.")
            sys.exit(1)

        results = []
        for item in sorted(submissions_dir.iterdir()):
            if item.is_dir() and (item / "agent.py").exists():
                print(f"--- Evaluating {item.name} ---")
                r = evaluate_single_submission(item, lab_id, episodes, args.seed,
                                               isolated=args.isolated, sandbox_image=args.sandbox_image)
                results.append(r)
                print(f"Result: {r.get('mean_reward', 'ERR')} ± {r.get('std_reward', 0)} ({r.get('status')})\n")

        # Sort by mean reward descending
        results.sort(key=lambda x: x.get("mean_reward", -9999), reverse=True)

        if args.output_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
