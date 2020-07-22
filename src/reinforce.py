import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import skimage.transform
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('MsPacman-v0')
gpu = torch.device('cuda:1')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 2)
        self.conv4 = nn.Conv2d(32, 16, 3, 2)
        self.affine1 = nn.Linear(16*14*14, 256)
        self.affine2 = nn.Linear(256, 9)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(-1, 16*14*14)
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


lr = 0.000125
print('learning rate:', lr)
policy = Policy().to(gpu)
optimizer = optim.Adam(policy.parameters(), lr=lr)
eps = np.finfo(np.float32).eps.item()


def resize(image, size):
    return skimage.transform.resize(image, (size, size))


def select_action(state):
    state = state.transpose(2, 0, 1)
    state = torch.tensor(state, device=gpu).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    future_steps = 0
    for r in policy.rewards[::-1]:
        # if r > 0:
        #     R = 0
        future_steps += 1
        R = r + args.gamma * R
        returns.insert(0, R / future_steps)
    returns = torch.tensor(returns, device=gpu)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    avg_rewards = []
    avg_reward = 0
    max_reward = 0
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        # prev_state = resize(state, 64)
        for t in range(1, 10000):  # Don't infinite loop while learning
            # state = resize(state, 64) - prev_state
            # prev_state = state
            state = resize(state, 64)

            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        avg_reward += ep_reward

        finish_episode()
        if i_episode % args.log_interval == 0:
            avg_reward = avg_reward / args.log_interval
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, avg_reward))

            avg_rewards.append(avg_reward)
            avg_reward = 0

            # save the best model
            if avg_reward > max_reward:
                max_reward = avg_reward
                torch.save(policy.state_dict(), 'policy.pt')

            # plot a graph
            plt.clf()
            x = np.arange(args.log_interval, i_episode + 1, args.log_interval)
            plt.plot(x, avg_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.savefig("graph.png")


if __name__ == '__main__':
    main()
