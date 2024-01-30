# Evaluating Long-Term Time-Series Forecasting Methods for Network Traffic Prediction

Python 3.6
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)

This is the origin Pytorch implementation of the following paper:
[Evaluating Long-Term Time Series Forecasting Methods for Network Traffic Prediction](Link)

## Content

:triangular_flag_on_post: Preprocessing of network traffic.

:triangular_flag_on_post: STFT_Framework model.

:triangular_flag_on_post: Slurm configuration to run different models on the dataset.

:triangular_flag_on_post: Includes Nonstationary_Transformers, Informer, PatchTST, NLinear, DLinear, RLinear, Linear.

## Data

The dataset was taken from the paper [Network Traffic Characteristics of Data Centers in the Wild](https://pages.cs.wisc.edu/~tbenson/IMC10_Data.html). Only the UNI1 packet traces dataset was used. However the other packet traces datasets (UNI1, PRV1) or any packet traces saved in a pcap-file can be used with the introduced preprocessing methods.

## Preprocessing

1. Download packet trace dataset from [the website](https://pages.cs.wisc.edu/~tbenson/IMC10_Data.html)
2. Save pcap-files in the data repository (optionally in another subdirectory).
3. Configure parameter in the data_provider/data_preparer main-method (e.g. path/to/pcapFiles and so on).
4. Run data_provider/data_preparer main-method - Now the resulting flows (aggregated) should be saved in data directory.
5. Search in the conf/\* directory for the wanted configuration and update parameter like paths (!!) to the location where the data was saved in the last step (4).
6. Optionally change hyperparameters
7. Run

```
python LtsfExperiment.py -o <path/to/config.yaml>
```

## Potential Errors

- Wrong Paths in (be careful with / and \\) config or data_preparer
- Wandb has to be configured

## Acknowledgemends:

Parts of the code were taken from the following repositories:

- https://github.com/thuml/Autoformer (Vanilla Transformer, Informer, and general structures)
- https://github.com/PatchTST/PatchTST (PatchTST)
- https://github.com/thuml/Nonstationary_Transformers (Non-Stationary Transformer)
- https://github.com/cure-lab/LTSF-Linear (DLinear and NLinear)
- https://github.com/plumprc/RTSF (RLinear)

Thanks for the computing infrastructure provided by the Scientific Computing Center [(SCC)](https://www.scc.kit.edu/).
