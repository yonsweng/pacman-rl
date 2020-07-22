import argparse
import torch
from src import dqn, reinforce

parser = argparse.ArgumentParser(description='train-gpu1')
parser.add_argument('--dqn', default=False, action='store_true')
parser.add_argument('--lr', default=0.001, type=float, metavar='f', help='learning rate')
parser.add_argument('--log-interval', default=10, type=int, metavar='i', help='log interval')
parser.add_argument('--gamma', default=0.99, type=float, metavar='f', help='gamma')
parser.add_argument('--seed', default=543, type=int, metavar='i', help='seed')
args = parser.parse_args()

device = torch.device('cuda:1')

if args.dqn:
    dqn.train(device, args)
else:
    reinforce.train(device, args)
