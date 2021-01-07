# RR dataset

## RR-passage
In RR-passage dataset, all argument pairs from the same passage pair are put into only one of the training, development and testing sets. 

## RR-submission
However, different review-rebuttal passage pairs of the same submission could be put into different sets.
Since different reviewers may discuss similar issues for one submission, different review-rebuttal passage pairs of the same submission may share similar context information.
To alleviate this effect, we also prepare another dataset version split on the submission level, namely RR-submission.
In RR-submission, multiple review-rebuttal passage pairs of the same submission are in the same set.

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

## Data Access
Kindly drop me an [email](liying.cheng@alibaba-inc.com) to access the data.

## Citation
```
@inproceedings{cheng2020ape,
  title={APE: Argument Pair Extraction from Peer Review and Rebuttal via Multi-task Learning},
  author={Cheng, Liying and Bing, Lidong and Qian, Yu and Lu, Wei and Si, Luo},
  booktitle={Proceedings of EMNLP},
  year={2020}
}
```
