# MY_CW_MAIN.py
from cw2.cw_data.cw_wandb_logger import WandBLogger

import wandb
from cw2 import cluster_work
from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from torch import optim, nn

from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from utils.tools import dotdict


class LtsfExperiment(experiment.AbstractIterativeExperiment):
    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        cw_logging.getLogger().info("Ready to start repetition {}. Resetting everything: {}".format(rep, cw_config))

        params = cw_config['params']
        params['iterations'] = cw_config['iterations']

        self.config = dotdict(params)
        self.expMain = Exp_Main(self.config)

        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')

        self.criterion = self._select_criterion()
        self.model_optim = self._select_optimizer()

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        train_loss, trues_preds_train = self.expMain.train(n, train_data=self.train_data,
                                                           train_loader=self.train_loader,
                                                           criterion=self.criterion,
                                                           model_optim=self.model_optim)  # train step
        vali_loss, trues_preds_vali = self.expMain.vali(vali_data=self.vali_data, vali_loader=self.vali_loader,
                                                        criterion=self.criterion)  # vali
        test_loss, trues_preds_test = self.expMain.vali(vali_data=self.test_data, vali_loader=self.vali_loader,
                                                        criterion=self.criterion)  # test

        cw_logging.getLogger().info(f"epoch: {n} | train loss: {train_loss} | vali loss: {vali_loss} | test loss: {test_loss}")
        results = {"test_loss": test_loss, "vali_loss": vali_loss, "train_loss": train_loss, 'iter': n}

        # log results as diagrams
        if n + 1 == cw_config['iterations']:
            self.__log_trues_preds__(cw_config, trues_preds_train, "train")
            self.__log_trues_preds__(cw_config, trues_preds_test, "test")
            self.__log_trues_preds__(cw_config, trues_preds_vali, "vali")

            test_results, _ = self.expMain.test(test_data=self.test_data, test_loader=self.test_loader)  # get metrics
            results.update(test_results)

        return results

    def __log_trues_preds__(self, cw_config, trues_preds: list, title: str):
        cw_logging.getLogger().info(f"Saving results {title} as plots in wandb...")
        xs = [i for i in range(cw_config['params']['seq_len'] + 1,
                               cw_config['params']['seq_len'] + 1 + cw_config['params']['pred_len'])]
        y, y_pred = trues_preds[-1]

        for i in range(y.size(0)):
            y_p, y_pred_p = y[i, :, 0].detach().cpu().tolist(), y_pred[i, :, 0].detach().cpu().tolist()
            wandb.log({f"{title}_ground_truth_prediction_{i}": wandb.plot.line_series(xs=xs, ys=[y_p, y_pred_p],
                                                                                      keys=['Ground Truth',
                                                                                            'Prediction'],
                                                                                      title=f"{title} Ground Truth vs. Prediction")})

        cw_logging.getLogger().info(f"Finished saving diagrams {title} in wandb!")

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        if surrender is not None:
            cw_logging.getLogger().info("Run was surrendered early.")
            return

        if crash:
            cw_logging.getLogger().warning("Run crashed with an exception.")
            return

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.config, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.expMain.model.parameters(), lr=self.config.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(LtsfExperiment)  # wrap_experiment()
    # create_sweep(cw)
    cw.add_logger(WandBLogger())

    # RUN!
    cw.run()
