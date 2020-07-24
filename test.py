import gym
import argparse
import torch
import time
import numpy as np
from src.dqn import DQN
import cv2


def pre_proc(X):
    x = np.mean(X, axis=2)  # rgb->gray
    x = np.uint8(cv2.resize(x, (RESIZED_SIZE, RESIZED_SIZE),
                            interpolation=cv2.INTER_LINEAR))
    return x


def get_action(q, e):
    if e > np.random.rand(1):
        action = np.random.randint(OUTPUT)
    else:
        action = np.argmax(q)
    return action


if __name__ == '__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--model', required=True, type=int)
    args = parser.parse_args()

    env = gym.make('MsPacman-v0')

    # Constants
    OUTPUT = env.action_space.n  # 9
    RESIZED_SIZE = 84
    HISTORY_SIZE = 4
    eps = 0.

    mainDQN = DQN(HISTORY_SIZE, OUTPUT)
    mainDQN.load_state_dict(torch.load(f'models/{args.model}.pt',
                            map_location='cpu'))
    mainDQN.eval()

    while True:
        history = np.zeros([HISTORY_SIZE+1, RESIZED_SIZE, RESIZED_SIZE],
                           dtype=np.uint8)
        state = env.reset()
        prev_state = pre_proc(state)
        history[:, :, :HISTORY_SIZE] = np.zeros((84, 84, HISTORY_SIZE))

        while True:
            env.render()

            # 히스토리의 0~3 부분으로 Q값 예측
            X = history[:, :, :4].transpose(2, 0, 1).reshape(-1, 4, 84, 84)/255
            X = torch.tensor(X).float()
            Q = mainDQN(X)
            Q = Q.reshape(-1).cpu().detach().numpy()
            action = get_action(Q, eps)

            state, reward, done, _ = env.step(action)

            # Save history in replay_memory
            pre_proc_s1 = pre_proc(state)
            history[:, :, 4] = pre_proc_s1 - prev_state
            prev_s = pre_proc_s1
            history[:, :, :4] = history[:, :, 1:]

            time.sleep(0.02)

            if done:
                break
