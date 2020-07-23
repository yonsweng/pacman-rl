import gym
import copy
import torch
import torch.nn as nn
import random
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from skimage.transform import resize
from src.dqn import DQN


def preprocess(X):
    # Resize
    return np.uint8(resize(X / 255, (RESIZED_SIZE, RESIZED_SIZE),
                    mode='reflect') * 255)


def select_action(Q):
    if eps > np.random.rand(1):
        return np.random.randint(N_ACTIONS)
    else:
        return np.argmax(Q)


def train(batch):
    s_stack = []
    a_stack = []
    r_stack = []
    s1_stack = []
    d_stack = []

    for s_r, a_r, r_r, d_r in batch:
        s_stack.append(s_r[0])
        a_stack.append(a_r)
        r_stack.append(r_r)
        s1_stack.append(s_r[1])
        d_stack.append(d_r)

    r_stack = np.array(r_stack, dtype='float32')
    d_stack = np.array(d_stack, dtype='float32')
    s_stack = np.array(s_stack, dtype='float32') / 255
    s_stack = s_stack.transpose(0, 3, 1, 2)
    s_stack = torch.tensor(s_stack, device=device)
    s1_stack = np.array(s1_stack, dtype='float32') / 255
    s1_stack = s1_stack.transpose(0, 3, 1, 2)
    s1_stack = torch.tensor(s1_stack, device=device)

    Q1 = targetDQN(s1_stack).detach().cpu().numpy().max(axis=1)
    y = r_stack + (1 - d_stack) * DISCOUNT * Q1
    y = torch.tensor(y, device=device)

    Q = mainDQN(s_stack)
    one_hot = torch.tensor(np.eye(N_ACTIONS, dtype='float32')[a_stack],
                           device=device)
    q_val = torch.sum(torch.mul(Q, one_hot), axis=1)

    loss_fn = nn.MSELoss()
    loss = loss_fn(q_val, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--lr', default=0.00001, type=float)
    args = parser.parse_args()

    env = gym.make('MsPacman-v0')

    now = datetime.datetime.now()
    print(now)

    # Constants
    HOUR_MIN = str(now.hour * 10000 + now.minute * 100 + now.second)
    RESIZED_SIZE = 84  # Input size of DQN
    N_ACTIONS = env.action_space.n
    START_EPS = 1.0
    FINAL_EPS = 0.05
    EXPLORATION_STEPS = 200000
    MEMORY_SIZE = 50000  # Steps
    MAINDQN_UPDATE_CYCLE = 4
    TARGETDQN_UPDATE_CYCLE = 20000  # Steps
    PLOT_CYCLE = 20
    TRAIN_START = 10000  # Steps
    BATCH_SIZE = 32
    DISCOUNT = 0.99

    device = torch.device(f'cuda:{args.gpu}')

    mainDQN = DQN(3, N_ACTIONS).to(device)
    targetDQN = copy.deepcopy(mainDQN)  # On device

    optimizer = torch.optim.Adam(mainDQN.parameters(), lr=args.lr)

    # Training-wide variables
    eps = START_EPS
    episode = 0
    episode_rewards = []
    avg_rewards = []
    step = 0
    max_reward = 0
    replay_memory = deque(maxlen=MEMORY_SIZE)

    while True:  # Loop several episodes
        # Episode-wide variables
        episode += 1
        episode_reward = 0
        history = np.zeros([2, RESIZED_SIZE, RESIZED_SIZE, 3],
                           dtype=np.uint8)

        state = env.reset()
        state = preprocess(state)
        history[0] = state

        while True:  # Until an episode ends
            # Step-wide variables
            step += 1

            # Select an action
            X = history[0].transpose(2, 0, 1).astype('float32') / 255
            X = torch.tensor(X, device=device)
            X = X.unsqueeze(0)
            Q = mainDQN(X).tolist()[0]
            action = select_action(Q)

            state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Save history in replay_memory
            state = preprocess(state)
            history[1] = state
            replay_memory.append((history.copy(),
                                 action, reward, done))
            history[0] = history[1]

            # Decrease the exploration rate
            if eps > FINAL_EPS:
                eps -= (START_EPS - FINAL_EPS) / EXPLORATION_STEPS

            # Update mainDQN
            if step % MAINDQN_UPDATE_CYCLE == 0 and step >= TRAIN_START:
                batch = random.sample(replay_memory, BATCH_SIZE)
                train(batch)

            # Update targetDQN
            if step % TARGETDQN_UPDATE_CYCLE == 0:
                targetDQN = copy.deepcopy(mainDQN)
                print(f'step: {step} | targetDQN updated')

            if done:
                break

        episode_rewards.append(episode_reward)

        if episode % PLOT_CYCLE == 0:
            avg_reward_per_episode = sum(episode_rewards)/len(episode_rewards)
            avg_rewards.append(avg_reward_per_episode)
            episode_rewards = []

            # Plot the graph
            plt.clf()
            plt.xlabel('Episode')
            plt.ylabel('Average reward per epiosde')
            plt.plot(range(PLOT_CYCLE, episode+1, PLOT_CYCLE), avg_rewards)
            plt.savefig(f'graphs/{HOUR_MIN}.png')

            # Save the best model
            if avg_reward_per_episode > max_reward:
                max_reward = avg_reward_per_episode
                torch.save(mainDQN.state_dict(),
                           f'models/{HOUR_MIN}.pt')
                model_saved = True
            else:
                model_saved = False

            print(f'episode: {episode} |',
                  f'avg reward: {avg_reward_per_episode} |',
                  f'eps: {eps} |',
                  f'model saved: {model_saved}')
