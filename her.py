# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import pyrallis
from torch.utils.tensorboard import SummaryWriter
from bit_flipping_env import BitFlippingEnv
from replay_buffer import ReplayBuffer
from tqdm import tqdm

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda"
    """cpu or cuda"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Bit-flipping"
    """the id of the environment"""
    n_episode: int = 200000
    """the number of episode to train the agent"""
    learning_rate: float = 0.001
    """the learning rate of the optimizer"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.98
    """the discount factor gamma"""
    tau: float = 0.05
    """the target network update rate"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    epsilon_decay: float = 0.001
    """the decay of the epsilon greedy"""
    n_bits: int = 6
    """the size of state space"""
    eval_steps: int = 500
    """the fluency"""
    k_future: int = 4
    """Her future state"""
    with_her: bool = True
    """using her or not"""
    her_strategy: str = "future"
    """the strategy of her"""
    checkpoints_path: Optional[str] = None
    """save model"""
    reward_shaping: bool = False
    """using reward shaping or not"""

@torch.no_grad()
def evaluate_policy(env, q_network, num_episodes=100, max_steps=None):
    q_network.eval()
    success_count = 0
    if max_steps is None:
        max_steps = env.max_steps

    for _ in range(num_episodes):
        obs_dict = env.reset()
        obs = obs_dict['observation']
        g = obs_dict['desired_goal']
        total_reward = 0
        done = False
        while not done:
            inputs = np.concatenate([obs, g])
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(args.device)
            q_values = q_network(inputs)
            action = torch.argmax(q_values, dim=-1).detach().cpu().numpy()  # Ensure to get action index
            
            next_obs_dict, reward, done, _, _ = env.step(action)
            total_reward += reward
            obs = next_obs_dict['observation']
        
        if (obs == g).all():
            success_count += 1
    
    success_rate = success_count / num_episodes
    q_network.train()
    return success_rate

class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(QNetwork, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.hidden = nn.Linear(self.n_inputs, 256)
        nn.init.kaiming_normal_(self.hidden.weight)
        self.hidden.bias.data.zero_()

        self.output = nn.Linear(256, self.n_outputs)
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.output(x)

def choose_action(inputs, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        q_values = q_network(inputs)
        action = torch.argmax(q_values, dim=1).detach().cpu().numpy()

    return action

if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}_{args.exp_name}_{args.seed}_{args.n_bits}_{args.her_strategy}_{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # env setup
    env = BitFlippingEnv(args.n_bits, continuous=False, reward_shaping=args.reward_shaping)
    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    q_network = QNetwork(n_inputs=2 * args.n_bits, n_outputs=args.n_bits).to(args.device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(n_inputs=2 * args.n_bits, n_outputs=args.n_bits).to(args.device)
    target_network.load_state_dict(q_network.state_dict())

    replay_buffer = ReplayBuffer(
        state_dim = args.n_bits,
        action_dim = 1,
        goal_dim = args.n_bits,
        buffer_size = args.buffer_size,
        device = args.device,
    )

    # saving config to the checkpoint
    if args.checkpoints_path is not None:
        print(f"Checkpoints path: {args.checkpoints_path}")
        os.makedirs(args.checkpoints_path, exist_ok=True)
        with open(os.path.join(args.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(args, f)

    # training
    start_time = time.time()
    global_step = 0
    count = 0
    epsilon = args.start_e
    for episode_num in tqdm(range(args.n_episode), desc='episodes'):
        observation = env.reset()
        state = observation['observation']
        goal = observation['desired_goal']
        episode = []
        done = False
        while not done:
            inputs = np.concatenate([state, goal])
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(args.device)
            action = choose_action(inputs, epsilon=epsilon)
            next_observation, reward, done, truncated, info = env.step(action)
            next_state = next_observation['observation']
            replay_buffer.add_transition(state, action, reward, next_state, done, goal)
            episode.append((state.copy(), action, reward, done, next_state.copy()))
            state = next_state

        # HER
        if args.with_her:
            for i, transition in enumerate(episode):
                state, action, reward, done, next_state = transition
                if args.her_strategy == 'final':
                    new_goal = episode[-1][-1]
                elif args.her_strategy == 'future':
                    future_transitions = random.choices(episode[i:], k=args.k_future)
                    new_goal = [t[-1] for t in future_transitions]
                elif args.her_strategy == 'episode':
                    new_goal = random.choices([t[-1] for t in episode], k=args.k_future)
                elif args.her_strategy == 'random':
                    new_goal = replay_buffer.sample_random_goals(args.k_future)
                else:
                    raise ValueError("Unknown HER strategy")
                
                if isinstance(new_goal, list):
                    for goal in new_goal:
                        if np.sum(next_state == goal) == args.n_bits:
                            reward = 0
                        else:
                            reward = -1
                            if args.reward_shaping:
                                distance = np.linalg.norm(next_state - goal, axis=-1)
                                reward = -(distance).astype(np.float32)
                        replay_buffer.add_transition(state, action, reward, next_state, done, goal)
                else:
                    if np.sum(next_state == new_goal) == args.n_bits:
                        reward = 0
                    else:
                        reward = -1
                        if args.reward_shaping:
                            distance = np.linalg.norm(next_state - new_goal, axis=-1)
                            reward = -(distance).astype(np.float32)
                    replay_buffer.add_transition(state, action, reward, next_state, done, new_goal)

        epsilon = max(epsilon - args.epsilon_decay, args.end_e)
        if replay_buffer._size >= args.batch_size:
            batch = replay_buffer.sample(args.batch_size)
            batch = [b.to(args.device) for b in batch]
            (states, actions, rewards, next_states, dones, goals) = batch
            target_inputs = torch.cat((next_states, goals), dim = -1)
            with torch.no_grad():
                target_max, _ = target_network(target_inputs).max(dim=-1)
                td_target = rewards.flatten() + args.gamma * target_max * (1 - dones.flatten())
            inputs = torch.cat((states, goals), dim = -1)
            old_val = q_network(inputs).gather(1, actions.long()).squeeze()
            loss = F.mse_loss(td_target, old_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                )
            
            # eval 
            if episode_num % args.eval_steps == 0:
                success_count = evaluate_policy(env, q_network, max_steps=args.n_bits)
                writer.add_scalar("eval/success_count", success_count, episode_num)
                writer.add_scalar("losses/td_loss", loss.detach().cpu().numpy(), episode_num)
                writer.add_scalar("losses/q_values", old_val.mean().item(), episode_num)
                writer.add_scalar("charts/SPS", int(episode_num / (time.time() - start_time)), episode_num)
                print('eval: ', success_count, loss.detach().cpu().numpy(), old_val.mean().item())

                # saving model param to the checkpoint
                if args.checkpoints_path is not None:
                    torch.save(
                        QNetwork.state_dict(),
                        os.path.join(args.checkpoints_path, f"{episode_num}.pt"),
                    )

    env.close()
    writer.close()



