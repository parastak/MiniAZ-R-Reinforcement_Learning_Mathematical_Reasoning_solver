# train_rl.py (Actor-Critic + GAE)
"""
this is new train_rl module which is based on Actor-critic policy
"""

# import necessary modules and libraries
import numpy as np
import torch
from agents.policy_model import build_actor_critic
from agents.learned_solver import extract_cofficients
from agents.proposer import generate_equation
from core.curriculum import CurriculumController
from core.environment import solve_equation
from utils import setup_logger, Task, task_to_log_line, save_task_to_json
import logging


setup_logger()
model, optimizer =build_actor_critic(lr=1e-4)
BATCH_SIZE = 64
NUM_EPISODES = 10000
GAMMA = 0.99
LAMBDA = 0.95
MODEL_SAVE_PATH = "logs/actor_critic_policy.pth"

#entropy schedule hyperparameters
ENTROPY_COEFF_MIN = 0.001
ENTROPY_COEFF_MAX = 0.05
ENTROPY_DECAY_RATE = 2500  # number of episodes over which to decay the bonus


curriculum = CurriculumController()


# GAE function
def compute_GAE(rewards, values, gamma=GAMMA, lam=LAMBDA):
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


def solve_ground_truth(eq: str) -> list[float]:
    res = solve_equation(eq)
    return res if isinstance(res, list) and res else [0.0]



def train():
    buffer = []
    all_tasks = []
    logging.info(f"Starting REINFORCE training with Curriculum for {NUM_EPISODES} episodes...")

    for episode in range(NUM_EPISODES):
        current_level = curriculum.get_level()
        equation = generate_equation(difficulty=current_level)
        features = extract_cofficients(equation)
        # features = normalize(add_noise(extract_cofficients(equation)))
        x = torch.tensor([features], dtype=torch.float32)


        action, log_prob, value, entropy = model(x)
        prediction = action.item()
        gt_solutions = solve_ground_truth(equation)

        # reward shaping based on absolute error
        error = min([abs(gt - prediction) for gt in gt_solutions])
        alpha = 0.2
        reward = np.exp(-alpha * error)

        # --- DETAILED CONSOLE LOGGING ---
        # Clear and easy-to-read printout for each step
        print("-" * 60)
        print(f"Episode: {episode} | Difficulty: {current_level.upper()}")
        print(f"  Equation:     {equation}")
        print(f"  Ground Truth: {[round(s, 2) for s in gt_solutions]}")
        print(f"  Prediction:   {prediction:.2f}")
        print(f"  Reward:       {reward:.3f}")
        print("-" * 60)

        curriculum.record_result(reward)

        buffer.append((x, log_prob, value, reward, entropy))

        task = Task(
            task_id=episode,
            equation_str=equation,
            reasoning_type='deduction',
            ground_truth_solution=gt_solutions,
            solver_output=[prediction],
            reward=reward,
            meta={
                "error": error,
                "log_std": model.actor_log_std.item(),
                "entropy": entropy.item()
            }
        )
        all_tasks.append(task)

        if len(buffer) >= BATCH_SIZE:
            states, log_probs, values, rewards, entropies = zip(*buffer)
            values = [v.item() for v in values]
            advantages = compute_GAE(rewards, values)
            returns = [a + v for a, v in zip(advantages, values)]

            #convert to tensor
            states_tensor = torch.cat(states)
            log_probs_tensor = torch.cat(log_probs)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1)
            returns_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
            entropy_tensor = torch.cat(entropies)

            #normalize advantages
            if advantages_tensor.std() > 0:
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

            #re-compute values for the policy update
            _, _, new_values, _ = model(states_tensor)

            # calculate losses
            policy_loss = -(log_probs_tensor * advantages_tensor).mean()
            value_loss = 0.5 * (new_values.unsqueeze(1) - returns_tensor).pow(2).mean()

            #dynamic entropy bonus
            current_entropy_coeff = ENTROPY_COEFF_MIN + (ENTROPY_COEFF_MAX - ENTROPY_COEFF_MIN) * np.exp(-episode / ENTROPY_DECAY_RATE)
            entropy_bonus = current_entropy_coeff * entropy_tensor.mean()

            # total loss
            loss = policy_loss + value_loss - entropy_bonus

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            logging.info(
                f"Episode {episode} | Difficulty: {current_level.upper()}  | Avg Reward: {np.mean(rewards):.3f} | Loss: {loss.item():.6f} | Entropy: {entropy_bonus.item():.4f} | log_std: {model.actor_log_std.item():.4f}")
            buffer = []

            if (episode + 1) % 1000 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode + 1,
                    'curriculum_level': curriculum.current_level_index
                }, MODEL_SAVE_PATH)
                logging.info(f"Checkpoint saved at episode {episode + 1}")

    save_task_to_json(all_tasks, 'logs/reinforce_curriculum2.json')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info("Training complete. Final model and log saved.")


if __name__ == "__main__":
    train()
