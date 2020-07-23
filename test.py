import gym
import argparse
import torch
import time
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from src.dqn import DQN


def preprocess(X):
    # Convert to grayscale and resize
    return np.uint8(resize(rgb2gray(X), (RESIZED_SIZE, RESIZED_SIZE),
                    mode='reflect') * 255)


def select_action(Q):
    if eps > np.random.rand(1):
        return np.random.randint(N_ACTIONS)
    else:
        return np.argmax(Q)


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--model', required=True, type=int)
    args = parser.parse_args()

    env = gym.make('MsPacman-v0')

    # Constants
    RESIZED_SIZE = 84
    HISTORY_SIZE = 4
    N_ACTIONS = env.action_space.n
    eps = 0.

    mainDQN = DQN(HISTORY_SIZE, N_ACTIONS)
    mainDQN.load_state_dict(torch.load(f'models/{args.model}.pt',
                            map_location='cpu'))

    while True:
        history = np.zeros([HISTORY_SIZE+1, RESIZED_SIZE, RESIZED_SIZE],
                           dtype=np.uint8)
        state = env.reset()
        state = preprocess(state)
        prev_state = state

        while True:
            env.render()

            # Select an action
            X = history[:HISTORY_SIZE, :, :].astype('float32') / 255
            X = torch.tensor(X)
            X = X.unsqueeze(0)
            Q = mainDQN(X).tolist()[0]
            action = select_action(Q)

            state, reward, done, _ = env.step(action)

            # Save history in replay_memory
            state = preprocess(state)
            state = state - prev_state
            prev_state = state
            history[HISTORY_SIZE, :, :] = state
            history[:HISTORY_SIZE, :, :] = history[1:, :, :]

            time.sleep(0.05)

            if done:
                break
