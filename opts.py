"""Argument parser"""

import argparse


def parse_opt():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--data_path', '--data_path', default='../SGRAF/',
                        help='path to datasets')
    parser.add_argument('--data_name', '--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', '--vocab_path', default='../SGRAF/vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--model_name', '--model_name', default='../SGRAF/runs/f30k_SGR/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', '--logger_name', default='../SGRAF/runs/f30k_SGR/log',
                        help='Path to save Tensorboard log.')

    # ----------------------- training setting ----------------------#
    parser.add_argument('--batch_size', '--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', '--num_epochs', default=500, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', '--lr_update', default=30, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--lr_decay_rate', '--lr_decay_rate', default=0.8, type=float,
                        help='The learning rate to decay in number of epochs.')
    parser.add_argument('--learning_rate', '--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', '--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', '--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', '--val_step', default=1000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', '--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--margin', '--margin', default=1., type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', '--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')

    # ------------------------- model setting -----------------------#
    parser.add_argument('--img_dim', '--img_dim', default=280, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--img_width', '--img_width', default=280, type=int,
                        help='Dimensionality of the image embedding width.')
    parser.add_argument('--word_dim', '--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', '--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', '--sim_dim', default=512, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', '--num_layers', default=3, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--dropout4text', '--dropout4text', default=0.0, type=float,
                        help='dropout rate of GRU layers.')
    parser.add_argument('--dropout4vsa', '--dropout4vsa', default=0.0, type=float,
                        help='dropout rate of visual similarity.')
    parser.add_argument('--dropout4tsa', '--dropout4tsa', default=0.0, type=float,
                        help='dropout rate of text similarity.')
    parser.add_argument('--bi_gru', '--bi_gru', default=True, action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', '--no_imgnorm', default=True, action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', '--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--module_name', '--module_name', default='SAF', type=str,
                        help='SGR, SAF')
    parser.add_argument('--sgr_step', '--sgr_step', default=3, type=int,
                        help='Step of the SGR.')

    opt = parser.parse_args()
    print(opt)
    return opt
