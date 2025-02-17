import torch
import argparse
import sys

from network_category_type_CL_transformer import RNNFactory

class Setting:
    
    ''' Defines all settings in a single place using a command line interface.
    '''
    
    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv]) # foursquare has different default args.
                
        parser = argparse.ArgumentParser()        
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        else:
            self.parse_gowalla(parser)        
        self.parse_arguments(parser)                
        args = parser.parse_args()
        
        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.rnn_factory = RNNFactory(args.rnn)
        self.is_lstm = self.rnn_factory.is_lstm()
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s
        self.alpha_start = args.alpha_start
        self.alpha_end = args.alpha_end
        self.decay_type = args.decay_type
        self.weight_type = args.weight_type
        self.loss_type = args.loss_type
        self.tau = args.tau
        self.mu = args.mu
        self.thre = args.thre
        self.decay_epoch = args.decay_epoch
        self.review = args.review
        self.poi_weight = args.poi_weight
        self.cate_weight = args.cate_weight
        self.type_weight = args.type_weight
        self.dataset =  args.dataset

        self.transformer_nhid = args.transformer_nhid
        self.transformer_dropout = args.transformer_dropout
        self.attention_dropout_rate = args.attention_dropout_rate
        self.transformer_nhead = args.transformer_nhead
        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.max_users = 0 
        self.sequence_length = 3
        self.min_checkins = 16
        self.batch_size = args.batch_size
        # evaluation        
        self.validate_epoch = args.validate_epoch
        self.report_user = args.report_user        
        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)        
    
    def parse_arguments(self, parser):        
        # training
        parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')        
        parser.add_argument('--hidden-dim', default=100, type=int, help='hidden dimensions to use')
        parser.add_argument('--lr', default = 0.0005, type=float, help='learning rate')
        parser.add_argument('--epochs', default=100, type=int, help='amount of epochs') 
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data, and zero means randomly initialize the value')
        parser.add_argument('--lambda_s', default=0.1, type=float, help='decay factor for spatial data, and zero means randomly initialize the value')
        parser.add_argument('--alpha_start', default=1, type=float)
        parser.add_argument('--alpha_end', default=0.1, type=float)
        parser.add_argument('--decay_type', default='linear', type=str)
        parser.add_argument('--weight_type', default='user_weight', type=str)
        parser.add_argument('--loss_type', default='POI_cate', type=str)
        parser.add_argument('--poi_weight', default=0.3, type=float)
        parser.add_argument('--cate_weight', default=0.3, type=float)
        parser.add_argument('--type_weight', default=0.3, type=float)
        parser.add_argument('--dataset', default='CHA.txt', type=str, help='the dataset under ./data/<dataset.txt> to load')                
        parser.add_argument('--validate-epoch', default=5, type=int, help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int, help='report every x user on evaluation (-1: ignore)')    
        parser.add_argument('--tau', default=0.8, type=float)
        parser.add_argument('--mu', default=0.5, type=float)
        parser.add_argument('--thre', default=0.0, type=float)    
        parser.add_argument('--decay_epoch', default=10, type=int)
        parser.add_argument('--review', default='0', help='review or not')
        parser.add_argument('--transformer_nhid',
                        type=int,
                        default=32,
                        help='Hid dim in TransformerEncoder')
        parser.add_argument('--transformer_dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for transformer')
        parser.add_argument('--attention_dropout_rate',
                        type=float,
                        default=0.1,
                        help='Dropout rate for attention_dropout_rate')
        parser.add_argument('--transformer_nhead',
                        type=int,
                        default=2,
                        help='Num of heads in multiheadattention')
    
    def parse_gowalla(self, parser):
        parser.add_argument('--batch-size', default=10, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--weight_decay', default=0.00025, type=float, help='weight decay regularization')
    def parse_foursquare(self, parser):
        parser.add_argument('--batch-size', default=1024, type=int, help='amount of users to process in one pass (batching)')
        parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay regularization')
    

        
