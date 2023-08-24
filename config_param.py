import torch

class Config(object):
    def __init__(self):
        self.main_path = './'   # the main directory path of code running
        
        self.valid_vs_total = 0.2   # the proportion of the valid dataset to the total dataset
        
        self.batch_size = 64
        self.n_epochs = 50
        
        self.init_lr = 1e-2
        self.lr_decay = 10
        
        self.early_stop = False
        self.patience = 10
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.data_dir = self.main_path + 'data/'     # the directory path of data
        self.label_dir = self.main_path + 'label/'   # the directory path of label
        self.predict_data_dir = self.main_path + 'predict_data/'     # the directory path of predict data
        self.predict_label_dir = self.main_path + 'predict_label/'   # the directory path of predict label
        self.save_dir = self.main_path + 'save/'     # the directory path of saving model & result
        self.fig_dir = self.main_path + 'figure/'    # the directory path of storing figure