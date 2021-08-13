import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import argparse

def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    args['seed'] = seed_list
    args['device'] = device
    _train(args)


def _train(args):
    longtail='Longtail' if args['longtail']==1 else 'Normal'
    logfilename = '{}_{}_{}_{}_{}_{}_{}'.format(args['seed'], args['model_name'], args['convnet_type'],
                                                args['dataset'], args['init_cls'], args['increment'],longtail)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(args)
    logging.info('Seed: {}'.format(args['seed']))
    logging.info('Model: {}'.format(args['model_name']))
    logging.info('Convnet: {}'.format(args['convnet_type']))
    logging.info('Dataset: {}'.format(args['dataset']))
    _set_device(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'],args['longtail'])
    model = factory.get_model(args['model_name'], args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    for task in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            #logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            #logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
            logging.info('NCM top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NCM top5 curve: {}\n'.format(nme_curve['top5']))
        else:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))


def _set_device(args):
    device_type = args['device']

    if device_type == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(device_type))

    args['device'] = device
