import argparse
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CELS example')
    parser.add_argument('--model', type=str, default='CELS', help='use model', choices=['CELS'])
    parser.add_argument('--dataset', type=str, default='criteo', help='use dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--mutation', type=int, default=1, help='use mutation: 1 use 0 not used')
    parser.add_argument('--strategy', type=str, default='n,1', help='use strategy', choices=['1,1', '1+1', 'n,1', 'n+1'])
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'criteo':
        from run.run_criteo_cels import train
    elif dataset == 'avazu':
        from run.run_avazu_cels import train
    elif dataset == 'huawei':
        from run.run_huawei_cels import train
    train(params=args)






