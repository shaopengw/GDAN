import argparse

parser = argparse.ArgumentParser(description='Training GNN')
'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--cuda', type=int, default=2,
                    help='Avaiable GPU ID')

'''
   Model arguments
'''
parser.add_argument('--conv_name', type=str, default='riskgnn',
                    choices=['gat', 'gcn','rgcn','hat','han','hetgnn','hgnn','ie-hgcn','hwnn','riskgnn','hgt','sehgnn'],
                    help='The name of GNN filter.')
parser.add_argument('--input_dim', type=int, default=128,
                    help='Number of input dimension')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='Number of hidden dimension')
parser.add_argument('--output_dim', type=int, default=32,
                    help='Number of output dimension')
parser.add_argument('--n_heads', type=int, default=2,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of HeteGAT layers')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout ratio')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')

parser.add_argument('--n_epoch', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--clip', type=float, default=1,
                    help='Gradient Norm Clipping')
parser.add_argument('--weight_decay', type=float, default=0.1,
                    help='weight decay of adamw ')

args = parser.parse_args()
