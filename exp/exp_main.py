from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, TransformerPytorch, MLPLinear, \
    RLinear, RMLP, TestLinear
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
            'TestLinear': TestLinear

        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        def _run_model():
            if 'RLinear' in self.args.model:
                outputs = self.model(batch_x, batch_y[:, -self.args.pred_len:, :])
            elif 'Linear' in self.args.model or 'TST' in self.args.model:
                outputs = self.model(batch_x)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        true_pred = []
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

                true_pred.append((true.numpy(), pred.numpy()))
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, true_pred

    def train(self, epoch: int, train_data, train_loader, criterion, model_optim):
        time_now = time.time()

        train_steps = len(train_loader)
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.iterations,
                                            max_lr=self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        iter_count = 0
        train_loss = []

        trues_preds = []

        self.model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            trues_preds.append((batch_y.detach().cpu().numpy(), outputs.detach().cpu().numpy()))

            if (i + 1) % 100 == 0:
                print("\t iters: {0} | loss: {1:.7f}".format(i + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.iterations - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if self.args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
            epoch + 1, train_steps, train_loss))

        if self.args.lradj != 'TST':
            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        return train_loss, trues_preds

    def test(self, test_data, test_loader, test=0, inverse_scale=False):
        if test:
            print('loading model')
            # TODO load model
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        trues_preds = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                trues_preds.append((batch_y, outputs))

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        results = {}

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))

        results.update({'test_mse': mse, 'test_mae': mae, 'test_rmse': rmse,
                        'test_mape': mape, 'test_mspe': mspe})

        if inverse_scale:
            preds_inverse = test_data.inverse_transform(preds)
            trues_inverse = test_data.inverse_transform(trues)

            preds_inverse = preds_inverse.reshape(-1, preds_inverse.shape[-2], preds_inverse.shape[-1])
            trues_inverse = trues_inverse.reshape(-1, trues_inverse.shape[-2], trues_inverse.shape[-1])

            mae, mse, rmse, mape, mspe = metric(preds_inverse, trues_inverse)
            results.update({'real_test_mse': mse, 'real_test_mae': mae, 'real_test_rmse': rmse,
                            'real_test_mape': mape, 'real_test_mspe': mspe})

            trues_preds = list(zip(trues_inverse, preds_inverse))

        return results, trues_preds

    def predict(self, pred_data, pred_loader, load=False):
        if load:
            pass
            # path = os.path.join(self.args.checkpoints, setting)
            # best_model_path = path + '/' + 'checkpoint.pth'
            # self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        return preds
