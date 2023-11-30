import os.path

from cw2 import cluster_work, cw_error, experiment
from cw2.cw_data import cw_logging
from cw2.cw_data.cw_wandb_logger import WandBLogger

from utils.analysis import analyse_flows
from utils.data_perperation import parse_pcap_to_list_n


class DataPreparation(experiment.AbstractExperiment):
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        cw_logging.getLogger().info("Ready to start repetition {}. Resetting everything.".format(rep))

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        cw_logging.getLogger().info("Starting ...")

        isUni1 = config['params']['which'] == 'uni1'

        root_path = config['params']['root_path'] if not isUni1 else config['params']['root_path_u1']
        file_paths = config['params']['file_paths'] if not isUni1 else config['params']['file_paths_u1']
        save_path = config['params']['save_path'] if not isUni1 else config['params']['save_path_u1']
        # save_analyze = config['params']['save_analyse_path'] if not isUni1 else config['params']['save_analyse_path_u1']

        for i in range(len(file_paths)):
            file_paths[i] = os.path.join(root_path, file_paths[i])

        if not os.path.exists(save_path):
            parse_pcap_to_list_n(file_paths, save_path)

        cw_logging.getLogger().info("Done!")

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        if surrender is not None:
            cw_logging.getLogger().info("Run was surrendered early.")

        if crash:
            cw_logging.getLogger().warning("Run crashed with an exception.")
        cw_logging.getLogger().info("Finished. Closing Down.")


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(DataPreparation)
    cw.add_logger(WandBLogger())
    cw.run()
