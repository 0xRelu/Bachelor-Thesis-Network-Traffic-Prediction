from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, TransformerPytorch, MLPLinear, \
    RLinear, RMLP
from models.ns_models import ns_Transformer
from utils.tools import adjust_learning_rate
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        self.train_data, self.train_loader = None, None

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'TransformerPytorch': TransformerPytorch,
            'Informer': Informer,
            'Non-Stationary': ns_Transformer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'RLinear': RLinear,
            'RMLP': RMLP,
            'Linear': Linear,
            'MLPLinear': MLPLinear,
            'PatchTST': PatchTST,

        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        # collate_fn=padded_collate if self.args.data == 'Traffic_Single' else None)  # remove this if not Traffic data
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()  # nn.MSELoss()  #
        return criterion

    def vali(self):
        vali_data, vali_loader = self._get_data(flag='val')

        trues_preds = []
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'RLinear' in self.args.model:
                            y_input = batch_y[:, -self.args.pred_len:, :]
                            outputs = self.model(batch_x, y_input)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'RLinear' in self.args.model:
                        y_input = batch_y[:, -self.args.pred_len:, :]
                        outputs = self.model(batch_x, y_input)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                trues_preds.append((batch_y, outputs))

                loss = self.criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, trues_preds

    def train(self, epoch: int):
        if self.train_data is None or self.train_loader is None:
            self.train_data, self.train_loader = self._get_data(flag='train')
        time_now = time.time()

        train_steps = len(self.train_loader)
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=self.model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.iterations,
                                            max_lr=self.args.learning_rate)

        iter_count = 0
        train_loss = []

        trues_preds = []

        self.model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
            iter_count += 1
            self.model_optim.zero_grad()

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        y_input = batch_y[:, -self.args.pred_len:, :]
                        outputs, _ = self.model(batch_x, y_input)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self.criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if 'RLinear' in self.args.model:
                    y_input = batch_y[:, -self.args.pred_len:, :]
                    outputs = self.model(batch_x, y_input)
                elif 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp,
                                             batch_y_mark)  # before batch_y as 5th input
                # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                trues_preds.append((batch_y, outputs))

                loss = self.criterion(batch_y, outputs)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\t iters: {0} | loss: {1:.7f}".format(i + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.iterations - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if self.args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.model_optim)
                scaler.update()
            else:
                loss.backward()
                self.model_optim.step()

            if self.args.lradj == 'TST':
                adjust_learning_rate(self.model_optim, scheduler, epoch + 1, self.args, printout=False)
                scheduler.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss_avg = np.average(train_loss)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
            epoch + 1, train_steps, train_loss_avg))

        if self.args.lradj != 'TST':
            adjust_learning_rate(self.model_optim, scheduler, epoch + 1, self.args)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        return train_loss_avg, trues_preds

    def test(self):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        inputx = []

        trues_preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'RLinear' in self.args.model:
                            y_input = batch_y[:, -self.args.pred_len:, :]
                            outputs, _ = self.model(batch_x, y_input)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'RLinear' in self.args.model:
                        y_input = batch_y[:, -self.args.pred_len:, :]
                        outputs = self.model(batch_x, y_input)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                trues_preds.append((batch_y, outputs))

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

        if self.args.test_flop:
            #test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.array(preds)
        trues = np.array(trues)
        # inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        return {'mse': mse, 'mae': mae, 'rse': rse, 'rmse': rmse, 'mape': mape, 'mspe': mspe, 'corr': corr}, trues_preds

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'RLinear' in self.args.model:
                            y_input = batch_y[:, -self.args.pred_len:, :]
                            outputs, _ = self.model(batch_x, y_input)
                        elif 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'RLinear' in self.args.model:
                        y_input = batch_y[:, -self.args.pred_len:, :]
                        outputs, _ = self.model(batch_x, y_input)
                    elif 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
