import argparse
import torch
from src import dqn

# arguments parsing
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--gpu', required=True, type=int, metavar='i', help='cuda:i')
parser.add_argument('--lr', default=0.001, type=float, metavar='f', help='learning rate')
args = parser.parse_args()

# gpu
device = torch.device(f'cuda:{args.gpu}')
