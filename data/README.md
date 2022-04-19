# RR dataset

## RR-passage
In RR-passage dataset, all argument pairs from the same passage pair are put into only one of the training, development and testing sets. 

## RR-submission
However, different review-rebuttal passage pairs of the same submission could be put into different sets.
Since different reviewers may discuss similar issues for one submission, different review-rebuttal passage pairs of the same submission may share similar context information.
To alleviate this effect, we also prepare another dataset version split on the submission level, namely RR-submission.
In RR-submission, multiple review-rebuttal passage pairs of the same submission are in the same set.

## RR-submission-v2
In our [ACL 2021's work](https://aclanthology.org/2021.acl-long.496.pdf), we further modify the RR-Submission dataset by fixing some minor bugs in the labels, and name it RR-Submission-v2.
#### We suggest to use RR-submission-v2 in the future.

In ```train/dev/test.txt```, each line has 5 columns separate by ```\t```, as listed below:
* sentence
* B-Review / I-Review / B-Reply / I-Reply / O
* B-index / I-index / O
* Review / Reply
* Submission ID: defined by ourselves, not original ID from openreview


## Citation
```
@inproceedings{cheng2020ape,
  title={APE: Argument Pair Extraction from Peer Review and Rebuttal via Multi-task Learning},
  author={Cheng, Liying and Bing, Lidong and Qian, Yu and Lu, Wei and Si, Luo},
  booktitle={Proceedings of EMNLP},
  year={2020}
}
```
