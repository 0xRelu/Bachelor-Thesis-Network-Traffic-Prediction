
This is the codebase for the paper:
Parts of the code were taken from the following repositories:
- https://github.com/thuml/Autoformer (Vanilla Transformer, Informer, and general structures)
- https://github.com/PatchTST/PatchTST (PatchTST)
- https://github.com/thuml/Nonstationary_Transformers (Non-Stationary Transformer)
- https://github.com/cure-lab/LTSF-Linear (DLinear and NLinear)
- https://github.com/plumprc/RTSF (RLinear)

---

# Run application
0. To parse pcap-file to python object used by the DataPreparation: /utils/data_preparation_tools.py run parse_pcap_to_list_n
1. After that create flow time series: run /data_provider/data_preparer.py with right path
2. After that, choose the correct configuration from the conf folter and run: python LtsfExperiment.py -o <yourconfig.yaml>
