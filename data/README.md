# RR dataset

## RR-passage
In RR-passage dataset, all argument pairs from the same passage pair are put into only one of the training, development and testing sets. 

## RR-submission
However, different review-rebuttal passage pairs of the same submission could be put into different sets.
Since different reviewers may discuss similar issues for one submission, different review-rebuttal passage pairs of the same submission may share similar context information.
To alleviate this effect, we also prepare another dataset version split on the submission level, namely RR-submission.
In RR-submission, multiple review-rebuttal passage pairs of the same submission are in the same set.



## Citation
```
@inproceedings{cheng2020ape,
  title={APE: Argument Pair Extraction from Peer Review and Rebuttal via Multi-task Learning},
  author={Cheng, Liying and Bing, Lidong and Qian, Yu and Lu, Wei and Si, Luo},
  booktitle={Proceedings of EMNLP},
  year={2020}
}
```
