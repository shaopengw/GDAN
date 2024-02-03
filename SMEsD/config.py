import argparse

parser = argparse.ArgumentParser(description='Training GNN')
'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')

'''
   Model arguments
'''
parser.add_argument('--conv_name', type=str, default='riskgnn',
                    choices=['riskgnn'],
                    help='The name of GNN filter.')
parser.add_argument('--input_dim', type=int, default=23,
                    help='Number of input dimension')
parser.add_argument('--hidden_dim', type=int, default=18,
                    help='Number of hidden dimension')
parser.add_argument('--output_dim', type=int, default=12,
                    help='Number of output dimension')
parser.add_argument('--n_heads', type=int, default=2,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=2,
                    help='Number of HeteGAT layers')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout ratio')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')

parser.add_argument('--n_epoch', type=int, default=400,
                    help='Number of epoch to run')
parser.add_argument('--clip', type=float, default=1,
                    help='Gradient Norm Clipping')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='weight decay of adamw ')

args = parser.parse_args()

config={}
config['company_num']=3976