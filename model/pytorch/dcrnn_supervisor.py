import os
import time
import os.path as osp

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from lib import utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import masked_mae_loss
from lib.metrics import masked_mae_np, masked_rmse_np, masked_mape_np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}

class DCRNNSupervisor:
    def __init__(self, adj_mx, year, **kwargs):
        self.year = year
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, str(self.year) + 'info.log', level=log_level)

        self.num_nodes = int(adj_mx.shape[0])
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # data set
        self._data = utils.load_dataset(input_dim = self.input_dim, output_dim = self.output_dim, year = self.year, **self._data_kwargs)
        # self.standard_scaler = self._data['scaler']

        # setup model
        if (self.year > int(self._data_kwargs['begin_year'])):
            dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
            self.dcrnn_model = dcrnn_model.to(device) if torch.cuda.is_available() else dcrnn_model
            self.load_best_model()
            
        else :
            dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
            self.dcrnn_model = dcrnn_model.to(device) if torch.cuda.is_available() else dcrnn_model
        
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/' + str(self.year) + '/'):
            os.makedirs('models/' + str(self.year) + '/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/{year}/epo{epo_num}.tar'.format(year = self.year, epo_num = epoch))
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/{year}/epo{epo_num}.tar'.format(year = self.year, epo_num = self._epoch_num)), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/{year}/epo{epo_num}.tar'.format(year = self.year, epo_num = self._epoch_num), map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def load_best_model(self):
        self._setup_graph()
        epo_list = []
        for filename in os.listdir('models/{year}/'.format(year = int(self.year) - 1)): 
            epo_list.append(filename[3:]) 					# already has .tar in it
        epo_list= sorted(epo_list)
        load_path = 'models/{year}/epo{epo_num}'.format(year = int(self.year) - 1, epo_num = epo_list[-1])
        
        assert os.path.exists(load_path), 'Weights at {} not found'.format(load_path)
        checkpoint = torch.load(load_path)
        
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(load_path))


    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    #################################################

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []

            y_truths = []
            y_preds = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses)

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            # y_truths_scaled = []
            # y_preds_scaled = []
            # for t in range(y_preds.shape[0]):
                # y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                # y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                # y_truths_scaled.append(y_truth)
                # y_preds_scaled.append(y_pred)

            return mean_loss, {'prediction': y_preds, 'truth': y_truths}

    def metric(self, ground_truth, prediction):
        global result
        pred_time = [3,6,12]
        self._logger.info("[*] year {}, testing".format(self.year))
        for i in pred_time:
            mae = masked_mae_np(preds = prediction[:, :, :i], labels = ground_truth[:, :, :i], null_val=0)
            rmse = masked_rmse_np(preds = prediction[:, :, :i], labels = ground_truth[:, :, :i], null_val=0)
            mape = masked_mape_np(preds = prediction[:, :, :i], labels = ground_truth[:, :, :i], null_val=0)
            self._logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
            result[i]["mae"][self.year] = mae
            result[i]["mape"][self.year] = mape
            result[i]["rmse"][self.year] = rmse
        return mae

    def test_model(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []

            y_truths = []
            y_preds = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x)
                loss = self._compute_loss(y, output)
                
                losses.append(loss.item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses)

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            _ = self.metric(y_truths, y_preds)
            # y_truths_scaled = []
            # y_preds_scaled = []
            # for t in range(y_preds.shape[0]):
                # y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                # y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                # y_truths_scaled.append(y_truth)
                # y_preds_scaled.append(y_pred)

            return mean_loss, {'prediction': y_preds, 'truth': y_truths}

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

        self._logger.info(self.year)
        self._logger.info('Start training ...')
        
        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num
        use_time = []

        total_time = 0

        for epoch_num in range(self._epoch_num, epochs):

            self.dcrnn_model = self.dcrnn_model.train()

            train_iterator = self._data['train_loader'].get_iterator()
            losses = []

            start_time = datetime.now()

            for _, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x, y, batches_seen)

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
                
                # y = y.to('cpu')
                # output = output.to('cpu')

                loss = self._compute_loss(y_true = y.cpu(), y_predicted = output.cpu())

                self._logger.debug(loss.item())

                losses.append(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)

                optimizer.step()
            
            end_time = datetime.now()

            total_time += (end_time - start_time).total_seconds()

            use_time.append((end_time - start_time).total_seconds())

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now ! ")

            val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)

            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss, lr_scheduler.get_last_lr()[-1], float(use_time[-1]))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1: 
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), lr_scheduler.get_last_lr()[-1], float(use_time[-1]))
                test_loss, _ = self.test_model(dataset='test', batches_seen=batches_seen)

                self._logger.info(message)
            
            if epoch_num == epochs - 1:
                message = 'YEAR : {} \n Total training time is : {:.1f}s \n Average traning time is : {:.1f}s'.format(self.year, total_time, sum(use_time)/len(use_time))
                self._logger.info(message)				

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

    def _prepare_data(self, x, y):
        # print('Raw data x :',x.shape)
        # print('Raw data y :',y.shape)
        x, y = self._get_x_y(x, y)
        # print('Get data x :',x.shape)
        # print('Get data y :',y.shape)
        x, y = self._get_x_y_in_correct_dims(x, y)
        # print('Final data x :',x.shape)
        # print('Final data y :',y.shape)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        # y_true = self.standard_scaler.inverse_transform(y_true)
        # y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
