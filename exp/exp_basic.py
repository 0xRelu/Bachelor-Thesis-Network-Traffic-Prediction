import abc
import os
import torch

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    @abc.abstractmethod
    def _select_criterion(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _select_optimizer(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_data(self, flag):
        raise NotImplementedError

    @abc.abstractmethod
    def vali(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, epoch: int):
        raise NotImplementedError

    @abc.abstractmethod
    def test(self):
        raise NotImplementedError
