'''
baseline model:
    standard intra-atten
    share parameters by default
'''

import logging
import itertools
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np
import sys
from models.baseline_snli import *
from models.baseline_snli import encoder
from models.baseline_snli import LSTMTagger
import argparse
from models.snli_data import snli_data
from models.snli_data import w2v
from random import shuffle



def train(args):
	
    use_cuda = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    if args.max_length < 0:
        args.max_length = 9999

    # initialize the logger
    # create logger
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(args.log_dir + args.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # stream handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    torch.cuda.set_device(args.gpu_id)
    
    #check cuda
    logger.info('cuda?')
    logger.info(use_cuda)
    logger.info(dtype)
    
    for arg in vars(args):
        logger.info(str(arg) + ' ' + str(getattr(args, arg)))

    # load train/dev/test data
    # train data
    logger.info('loading data...')
    train_data = snli_data(args.train_file, args.max_length)
    train_batches = train_data.batches
    train_lbl_size = 3
    dev_data = snli_data(args.dev_file, args.max_length)
    dev_batches = dev_data.batches
    test_data = snli_data(args.test_file, args.max_length)
    test_batches = test_data.batches
    logger.info('train size # sent ' + str(train_data.size))
    logger.info('dev size # sent ' + str(dev_data.size))
    logger.info('test size # sent ' + str(test_data.size))

    # get input embeddings
    logger.info('loading input embeddings...')
    word_vecs = w2v(args.w2v_file).word_vecs 

    best_dev = []   # (epoch, dev_acc)

    # build the model
    input_encoder = encoder(word_vecs.size(0), args.embedding_size, args.hidden_size, args.para_init)
    input_encoder.embedding.weight.data.copy_(word_vecs)
    input_encoder.embedding.weight.requires_grad = False
    lstm = LSTMTagger(args.hidden_size, train_lbl_size)
    #inter_atten = atten(args.hidden_size, train_lbl_size, args.para_init)

    input_encoder.cuda()
    lstm.cuda()

    para1 = filter(lambda p: p.requires_grad, input_encoder.parameters())
    para2 = lstm.parameters()

    if args.optimizer == 'Adam':
        input_optimizer = optim.Adam(para1,  lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        lstm_optimizer = optim.Adam(para2,  lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adagrad':
        input_optimizer = optim.Adagrad(para1, lr=args.lr, weight_decay=args.weight_decay)
        lstm_optimizer = optim.Adagrad(para2, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        input_optimizer = optim.Adadelta(para1, lr=args.lr)
        lstm_optimizer = optim.Adadelta(para2, lr=args.lr)
    else:
        logger.info('No Optimizer.')
        sys.exit()

    criterion = nn.NLLLoss(size_average=True)
    # criterion = nn.CrossEntropyLoss()

    logger.info('start to train...')
    for k in range(args.epoch):

        total = 0.
        correct = 0.
        loss_data = 0.
        train_sents = 0.

        shuffle(train_batches)
        timer = time.time()
        for i in range(len(train_batches)):

            train_src_batch, train_tgt_batch, train_lbl_batch = train_batches[i]

            train_src_batch = Variable(train_src_batch.cuda())
            train_tgt_batch = Variable(train_tgt_batch.cuda())
            train_lbl_batch = Variable(train_lbl_batch.cuda())

            batch_size = train_src_batch.size(0)
            train_sents += batch_size

            input_optimizer.zero_grad()
            lstm_optimizer.zero_grad()
            
            
            # initialize the optimizer
            if k == 0 and optim == 'Adagrad':
                for group in input_optimizer.param_groups:
                    for p in group['params']:
                        state = input_optimizer.state[p]
                        state['sum'] += args.Adagrad_init
                for group in lstm_optimizer.param_groups:
                    for p in group['params']:
                        state = lstm_optimizer.state[p]
                        state['sum'] += args.Adagrad_init
            

            train_src_linear, train_tgt_linear = input_encoder(
                train_src_batch, train_tgt_batch)
            log_prob = lstm(train_src_linear, train_tgt_linear, p_dropout = args.dropout)

            loss = criterion(log_prob, train_lbl_batch)

            loss.backward()

            grad_norm = 0.
            para_norm = 0.

            #gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm(itertools.chain(lstm.parameters(),\
            filter(lambda p: p.requires_grad, input_encoder.parameters())), args.max_grad_norm)
           
            input_optimizer.step()
            lstm_optimizer.step()

            _, predict = log_prob.data.max(dim=1)
            total += train_lbl_batch.data.size()[0]
            correct += torch.sum(predict == train_lbl_batch.data)
            loss_data += (loss.data[0] * batch_size)  # / train_lbl_batch.data.size()[0])

            if (i + 1) % args.display_interval == 0:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, loss %.3f, grad-norm %.3f, time %.2fs, ' %
                            (k, i + 1, len(train_batches), correct / total,
                             loss_data / train_sents, grad_norm, time.time() - timer))
                train_sents = 0.
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.
            if i == len(train_batches) - 1:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, loss %.3f, grad-norm %.3f, time %.2fs, ' %
                            (k, i + 1, len(train_batches), correct / total,
                             loss_data / train_sents, grad_norm, time.time() - timer))
                train_sents = 0.
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.           

        # evaluate
        if (k + 1) % args.dev_interval == 0:
            input_encoder.eval()
            lstm.eval()
            correct = 0.
            total = 0.
            for i in range(len(dev_batches)):
                dev_src_batch, dev_tgt_batch, dev_lbl_batch = dev_batches[i]

                dev_src_batch = Variable(dev_src_batch.cuda())
                dev_tgt_batch = Variable(dev_tgt_batch.cuda())
                dev_lbl_batch = Variable(dev_lbl_batch.cuda())

                # if dev_lbl_batch.data.size(0) == 1:
                #     # simple sample batch
                #     dev_src_batch=torch.unsqueeze(dev_src_batch, 0)
                #     dev_tgt_batch=torch.unsqueeze(dev_tgt_batch, 0)

                #initialize hidden state
                
                #lstm.hidden = lstm.init_hidden(dev_src_batch.size()[0], dev_src_batch.size()[1])

                dev_src_linear, dev_tgt_linear=input_encoder(
                    dev_src_batch, dev_tgt_batch)
                log_prob=lstm(dev_src_linear, dev_tgt_linear)

                _, predict=log_prob.data.max(dim=1)
                total += dev_lbl_batch.data.size()[0]
                correct += torch.sum(predict == dev_lbl_batch.data)

            dev_acc = correct / total
            logger.info('dev-acc %.3f' % (dev_acc))

            if (k + 1) / args.dev_interval == 1:
                model_fname = '%s%s_epoch-%d_dev-acc-%.3f' %(args.model_path, args.log_fname.split('.')[0], k, dev_acc)
                torch.save(input_encoder.state_dict(), model_fname + '_input-encoder.pt')
                torch.save(lstm.state_dict(), model_fname + '_lstm.pt')
                best_dev.append((k, dev_acc, model_fname))
                logger.info('current best-dev:')
                for t in best_dev:
                    logger.info('\t%d %.3f' %(t[0], t[1]))
                logger.info('save model!') 
            else:
                if dev_acc > best_dev[-1][1]:
                    model_fname = '%s%s_epoch-%d_dev-acc-%.3f' %(args.model_path, args.log_fname.split('.')[0], k, dev_acc)
                    torch.save(input_encoder.state_dict(), model_fname + '_input-encoder.pt')
                    torch.save(lstm.state_dict(), model_fname + '_lstm.pt')
                    best_dev.append((k, dev_acc, model_fname))
                    logger.info('current best-dev:')
                    for t in best_dev:
                        logger.info('\t%d %.3f' %(t[0], t[1]))
                    logger.info('save model!') 
            '''
            early stopping if there has not been improvement in val accuracy for more than 10 epochs
            '''
            logger.info('Epochs since improvement in dev: %s' %(k-best_dev[-1][0]))
            if k-best_dev[-1][0] > 20:
                logger.info('Early stopping triggered')
                break

            input_encoder.train()
            lstm.train()

    logger.info('training end!')
    # test
    best_model_fname = best_dev[-1][2]


if __name__ == '__main__':
    parser=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_file', help='training data file (hdf5)',
                        type=str, default='/disk/scratch/bowenli/nmt/struct-attn/data/snli/baseline/entail-train.hdf5')

    parser.add_argument('--dev_file', help='development data file (hdf5)',
                        type=str, default='/disk/scratch/bowenli/nmt/struct-attn/data/snli/baseline/entail-val.hdf5')

    parser.add_argument('--test_file', help='test data file (hdf5)',
                        type=str, default='/disk/scratch/bowenli/nmt/struct-attn/data/snli/baseline/entail-test.hdf5')

    parser.add_argument('--w2v_file', help='pretrained word vectors file (hdf5)',
                        type=str, default='/disk/scratch/bowenli/nmt/struct-attn/data/snli/baseline/glove.hdf5')

    parser.add_argument('--log_dir', help='log file directory',
                        type=str, default='/disk/scratch/bowenli/nmt/struct-attn/data/snli/experiment_struc/')

    parser.add_argument('--log_fname', help='log file name',
                        type=str, default='log54.log')

    parser.add_argument('--gpu_id', help='GPU device id',
                        type=int, default=1)

    parser.add_argument('--embedding_size', help='word embedding size',
                        type=int, default=300)

    parser.add_argument('--epoch', help='training epoch',
                        type=int, default=250)

    parser.add_argument('--dev_interval', help='interval for development',
                        type=int, default=1)

    parser.add_argument('--optimizer', help='optimizer',
                        type=str, default='Adam')

    parser.add_argument('--Adagrad_init', help='initial accumulating values for gradients',
                        type=float, default=0.)

    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.05)

    parser.add_argument('--hidden_size', help='hidden layer size',
                        type=int, default=300)

    parser.add_argument('--dropout', help='dropout',
                        type=float, default=0.2)
                        
    parser.add_argument('--max_length', help='maximum length of training sentences,\
                        -1 means no length limit',
                        type=int, default=10)

    parser.add_argument('--display_interval', help='interval of display',
                        type=int, default=1000)

    parser.add_argument('--max_grad_norm', help='If the norm of the gradient vector exceeds this renormalize it\
                               to have the norm equal to max_grad_norm',
                        type=float, default=5)

    parser.add_argument('--para_init', help='parameter initialization gaussian',
                        type=float, default=0.01)

    parser.add_argument('--weight_decay', help='l2 regularization',
                        type=float, default=5e-5)

    parser.add_argument('--model_path', help='path of model file (not include the name suffix',
                        type=str, default='/disk/scratch/bowenli/nmt/struct-attn/data/snli/experiment_struc/')

    args=parser.parse_args()
    # args.max_lenght = 10   # args can be set manually like this
    train(args)

else:
    pass
