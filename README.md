## Code Reference
[pytorch_lstmcrf](https://github.com/allanj/pytorch_lstmcrf)

### Requirements
* Python >= 3.6 and PyTorch >= 0.4.1
* AllenNLP package (if you use ELMo)

If you use `conda`:

```bash
git clone https://github.com/allanj/pytorch_lstmcrf.git

conda create -n pt_lstmcrf python=3.7
conda activate pt_lstmcrf
# check https://pytorch.org for the suitable version of your machines
conda install pytorch=1.3.0 torchvision cudatoolkit=10.0 -c pytorch -n pt_lstmcrf
pip install tqdm
pip install termcolor
pip install overrides
pip install allennlp
```

### Usage
1. Put the Glove embedding file (`glove.6B.100d.txt`) under `data` directory (You can also use ELMo/BERT/Flair, Check below.) Note that if your embedding file does not exist, we just randomly initalize the embeddings.
2. Simply run the following command and you can obtain results comparable to the benchmark above.
    ```bash
    python trainer.py
    ```
    If you want to use your 1st GPU device `cuda:0` and train models for your own dataset with elmo embedding:
    ```
    python trainer.py --device cuda:0 --dataset YourData --context_emb elmo --model_folder saved_models
    ```

##### Training with your own data. 
1. Create a folder `YourData` under the data directory. 
2. Put the `train.txt`, `dev.txt` and `test.txt` files (make sure the format is compatible, i.e. the first column is words and the last column are tags) under this directory.  If you have a different format, simply modify the reader in `config/reader.py`. 
3. Change the `dataset` argument to `YourData` when you run `trainer.py`. 

## Data Preparation
The preprocessed RR dataset is saved in `./data`. For more details regarding the dataset, please refer to [RR](https://github.com/LiyingCheng95/ArgumentPairExtraction/tree/master/data).

### Data Processing
To process the data, we adopt [bert-as-service](https://github.com/hanxiao/bert-as-service) as a tool to obtain the embeddings for all tokens [x0, x1, · · · , xT −1] in the sentence.

#### Install
```
pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`
```

#### Download a pre-trained BERT model
e.g. Download a [model](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip), then uncompress the zip file into some folder, say ```/tmp/english_L-12_H-768_A-12/```

#### Start the BERT service
```bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -max_seq_len NONE -pooling_strategy NONE```

#### Use Client to Get Sentence Encodes
Run ```../data_processing/dataProcessing.py```.

Now you will get ```vec_train.pkl```, ```vec_dev.pkl```, ```vec_test.pkl```.


## Citation
```
@inproceedings{cheng2020ape,
  title={APE: Argument Pair Extraction from Peer Review and Rebuttal via Multi-task Learning},
  author={Cheng, Liying and Bing, Lidong and Qian, Yu and Lu, Wei and Si, Luo},
  booktitle={Proceedings of EMNLP},
  year={2020}
}
```

