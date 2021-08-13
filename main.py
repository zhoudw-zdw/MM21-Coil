import json
import argparse
from trainer import train
import pprint


def main():
    _utils_pp = pprint.PrettyPrinter()
    args = setup_parser().parse_args()
    args = vars(args)
    _utils_pp.pprint(args)

    
    train(args)


def setup_parser():
    parser = argparse.ArgumentParser(description='Co-transport for Class-Incremental Learning')
    parser.add_argument('--prefix', type=str, default=' ')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--init_cls', type=int, default=20)
    parser.add_argument('--increment', type=int, default=20)
    parser.add_argument('--model_name', type=str, default='COIL')
    parser.add_argument('--convnet_type', type=str, default='cosine_resnet32')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--longtail', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1993)
    parser.add_argument('--sinkhorn', type=float, default=0.464)
    parser.add_argument('--calibration_term', type=float, default=1.5)
    parser.add_argument('--norm_term', type=float, default=3.)
    parser.add_argument('--reg_term',type=float,default=1e-3,help='Regularization term of backward transfering distillation loss')
    return parser


if __name__ == '__main__':
    main()
