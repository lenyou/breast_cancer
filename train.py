import numpy as np
from train.train_net import train
import private_config
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
    parser.add_argument('--pretrained', dest='pretrained', type=str, default="no_pretrain",
                        help='save training log to file')
    parser.add_argument('--epoch', dest='epoch', type=int, default="1",
                        help='yaml_config')
    args = parser.parse_args()
    return args

    
if __name__ == '__main__':
    args = parse_args()
    pretrained_path = args.pretrained
    pretrained_epoch = args.epoch
    train(private_config.train_path,private_config.batch_size,pretrained_path,pretrained_epoch)
    
    
