import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--N', type=int, default=4)
    return parser.parse_args()