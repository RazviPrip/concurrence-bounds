from plot_utils import plot_comparison, plot_heatmaps
from config import parse_args

def main():
    args = parse_args()
    plot_comparison(args)
    plot_heatmaps(args)

if __name__ == '__main__':
    main()