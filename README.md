
# This is the codebase for the paper:
Parts of the code were taken from the following repositories:
    - https://github.com/thuml/Autoformer (Vanilla Transformer, Informer, and general structures)
    - https://github.com/PatchTST/PatchTST (PatchTST)
    - https://github.com/thuml/Nonstationary_Transformers (Non-Stationary Transformer)
    - https://github.com/cure-lab/LTSF-Linear (DLinear and NLinear)
    - https://github.com/plumprc/RTSF (RLinear)

---

# Run application
1. When data not prepered: run /data_provider/DataPerperation.py -> main 
    - Configure seq_len, pred_len and step_size (= how many milliseconds should be skipped e.g if we have time series with 0-10 milliseconds and a step_size of 10 - only 0 and 10 are gonna be loaded)
2. Run `python LtsfExperiment.py -o ./conf/ltsf_transformer_config.yaml` (or any other configuration file as long as it is correctly configured. Not all of them are yet!)
    - be careful: The seq_len, pred_len should be equal to the one defined in DataPreperation
    - Furthermore, be careful with '\\' and '/' in the configuration files, when using Windows or Linux!
    - You should be logged in into Wandb or an error might be thrown
3. For quick test run without wandb and the cw2 environment: run LtsfExperimentTest.py (you can also configure the parameters in the directory there)
