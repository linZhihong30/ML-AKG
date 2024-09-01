import argparse
from engine import *
from models import *
from voc import *
from voc2012 import *

import os
from torch.utils.data.dataset import ConcatDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--dataset', '-d', default='voc2007', type=str,
                    metavar='D', help='image dataset type (default: voc2007)')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',  # epoch = 100
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[25, 50, 80], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,  # batch_size
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--seed', '--random-seed', default=42, type=int,
                    metavar='S', help='random seed')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main_voc2007():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    use_gpu = torch.cuda.is_available()

    # define dataset
    print('dataset type is {}'.format(args.dataset))
    if args.dataset == 'voc2007':

        train_dataset = Voc2007Classification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl')
        val_dataset = Voc2007Classification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl')
    elif args.dataset == 'voc2012':
        train_dataset = Voc2012Classification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl')
        val_dataset = Voc2007Classification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl')
    else:
        raise ValueError('The function expects voc2007 or voc2012')

    model_path = f'checkpoint/{args.dataset}/'

    num_classes = 20

    # load model
    model = gcn_clip(num_classes=num_classes, t=0.4, adj_file='data/voc/voc_adj.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,  # initial lr = 0.01
                                momentum=args.momentum,  # 0.9
                                weight_decay=args.weight_decay)  # 1e-4

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': num_classes, 'difficult_examples': True,
             'save_model_path': model_path, 'workers': args.workers, 'epoch_step': args.epoch_step,
             'lr': args.lr}
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    main_voc2007()
