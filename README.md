# Run application
1. When data not prepered: run /data_provider/DataPerperation.py -> main 
    - Configure seq_len, pred_len and step_size (= how many milliseconds should be skipped e.g if we have time series with 0-10 milliseconds and a step_size of 10 - only 0 and 10 are gonna be loaded)
2. Run `python LtsfExperiment.py -o ./conf/ltsf_transformer_config.yaml` (or any other configuration file as long as it is correctly configured. Not all of them are yet!)
    - be careful: The seq_len, pred_len should be equal to the one defined in DataPreperation
    - Furthermore, be careful with '\\' and '/' in the configuration files, when using Windows or Linux!
    - You should be logged in into Wandb or an error might be thrown
3. For quick test run without wandb and the cw2 environment: run LtsfExperimentTest.py (you can also configure the parameters in the directory there)

# Changes Worth Mentioning
1. DataPreperation and DataLoader: /data_provider/DataPreperation.py and /data_provider/data_loader.py
    - DataPreperation: Splits packets into sequences of format [X, K, 3] (K might be huge because of 
    padding we added, "3" contains packet features: actual time, size and direction) as input sequence and [X, pred_len, 2] 
    ("2" contains time in milliseconds and aggregated sizes of packets which arrived at that millisecond) as target sequence
    - DataLoader: Loads directory with both tensors {x,y} and creates stamps for temporal_embedding
2. Embeddings: /layers/Embed.py
    - TemporalEmbeddingMicroseconds: Gets [batch_size, K/pred_len, 2] ("2" contains milli- and microseconds) and embeds them
      - ![alt text](./img/TemporalEmbedding.png)
    - DirectionEmbedding: Gets [batch_size, K/pred_len, 1] (should contain direction)
      - ![alt text](./img/DirectionalEmbedding.png)
      - PROBLEM (i just realized): Only the input sequence carries direction for each packet -> so the target sequence 
      does not contain it. In the decoder when the embedding is called with the target sequence, the directionEmbedding 
      gets the size of the packets while the linear layer gets a 0 dim vector. (see picture DataEmbedding_w_temp - the slice in value_embedding/direction_embedding). 
      How to fix that? Should we add a direction to the target sequence (but there are multiple packets per millisecond potentially in different directions)
    - DataEmbedding_w_dir_temp: Finally the embedding we want to apply for ENCODER. Applies a Linear Layer (as value embedding), directional embedding, and Temporal Embedding
      - ![alt text](./img/DataEmbedding_w_dir_temp.png)
    - DataEmbedding_w_temp: This is the embedding we apply before the target sequence enters the DECODER. It only applies a Linear Layer (as value embedding) and a Temporal Embedding
      - ![alt text](./img/DataEmbedding_w_temp.png)
4. Masking: /utils/masking.py
   - PaddingMask: We added a custom self-attention mask, which should mask out the added paddings
     - ![alt text](./img/PaddingMask.png)
   - Not sure if this is worth mentioning: BUT this was followed by a small change in the vanilla Transformer architecture that it even allows to pass a custom mask (changed the boolean value of FullAttention to true):
     - ![alt text](./img/Transformer_Encoder.png)
   - I am NOT yet sure whether the mask does exactly what we want or not!! The values it produced seemed to be fine, BUT I got no certain evidence for that yet!