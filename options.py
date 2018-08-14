# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import argparse

parser = argparse.ArgumentParser()

##########################################################################################################
#                                               Data                                                     #
##########################################################################################################

parser.add_argument('--testDir', type = str, default = "data/test/",
                    help = 'Path to image for validation.')

##########################################################################################################
#                                               Train                                                    #
##########################################################################################################
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'Number of examples to process in a batch.')

parser.add_argument('--epoch', type = int, default = 200,
                    help = 'the number of epoch to train.')
parser.add_argument('--learning_rate', type = float, default = 1e-4,
                    help = 'The learning rate for optimizer.')
parser.add_argument('--train_log_freq', type = int, default = 100,
                    help = 'How often to log results to the console when training.')

parser.add_argument('--train_log_dir', type = str, default = "train_logs",
                    help = 'Directory where to write event logs and checkpoints.')

parser.add_argument('--train_from_exist', type = bool, default = False,
                    help = 'Whether to train model from pretrianed ones.')



##########################################################################################################
#                                                 test                                                  #
##########################################################################################################
parser.add_argument('--test_batch_size', type = int, default = 1,
                    help = 'Number of examples to process in a batch.')
parser.add_argument('--test_log_dir', type = str, default = 'test_logs/',
                    help = 'Directory where to write event logs.')

params = parser.parse_args()




