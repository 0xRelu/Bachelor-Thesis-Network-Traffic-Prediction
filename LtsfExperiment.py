# MY_CW_MAIN.py
from cw2.cw_data.cw_wandb_logger import WandBLogger

import wandb
from cw2 import cluster_work
from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from sweep_work.create_sweep import create_sweep
from sweep_work.experiment_wrappers import wrap_experiment
from sweep_work.sweep_logger import SweepLogger

from exp.exp_main import Exp_Main
from models.Config import Config


class LtsfExperiment(experiment.AbstractIterativeExperiment):
    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        cw_logging.getLogger().info("Ready to start repetition {}. Resetting everything: {}".format(rep, cw_config))

        params = cw_config['params']
        params['iterations'] = cw_config['iterations']

        config = Config(params)
        self.expMain = Exp_Main(config)

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        loss, trues_preds_train = self.expMain.train(n)  # train step
        test_dict, trues_preds_test = self.expMain.test()  # test
        vali_loss, trues_preds_vali = self.expMain.vali()  # vali

        test_dict.update({"vali_loss": vali_loss, "train_loss": loss, 'iter': n})

        # log results as diagramms
        if n + 1 == cw_config['iterations']:
            self.__log_trues_preds__(cw_config, trues_preds_train, "train")
            self.__log_trues_preds__(cw_config, trues_preds_test, "test")
            self.__log_trues_preds__(cw_config, trues_preds_vali, "vali")

        cw_logging.getLogger().info(f"epoch: {n} | loss: {loss}")
        return test_dict

    def __log_trues_preds__(self, cw_config, trues_preds: list, title: str):
        cw_logging.getLogger().info(f"Saving results {title} as plots in wandb...")
        xs = [i for i in range(cw_config['params']['seq_len'] + 1,
                               cw_config['params']['seq_len'] + 1 + cw_config['params']['pred_len'])]
        y, y_pred = trues_preds[-1]

        for i in range(y.size(0)):
            y_p, y_pred_p = y[i, :, 0].detach().cpu().tolist(), y_pred[i, :, 0].detach().cpu().tolist()
            wandb.log({f"{title}_ground_truth_prediction_{i}": wandb.plot.line_series(xs=xs, ys=[y_p, y_pred_p],
                                                                                      keys=['Ground Truth', 'Prediction'],
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


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(LtsfExperiment) #wrap_experiment()
    #create_sweep(cw)
    cw.add_logger(WandBLogger())

    # RUN!
    cw.run()
