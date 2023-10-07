# Multi-View-Co-Teaching-Network

This repository contains the source code for the CIKM 2020 paper **Learning to Match Jobs with Resumes from Sparse Interaction Data using Multi-View Co-Teaching Network**

## Directory

- [Motivations](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/README.md#Motivations)
- [Datasets](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/README.md#Datasets)
- [Download and Usage](https://github.com/Multi-View-Co-Teaching/blob/master/README.md#Download)
- [Licence](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/README.md#Licence)
- [References](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/README.md#References)
- [Additional Notes](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/README.md#Addition)

## Motivations

With the ever-increasing growth of online recruitment data, job-resume matching has become an important task to automatically match jobs with suitable resumes. This task is typically casted as a supervised text matching problem. Supervised learning is powerful when the labeled data is sufficient. However, on online recruitment platforms, job-resume interaction data is sparse and noisy, which affects the performance of job-resume match algorithms.
To alleviate these problems, in this paper, we propose a novel multi-view co-teaching network from sparse interaction data for job-resume matching. Our network consists of two major components, namely text-based matching model and relation-based matching model. The two parts capture semantic compatibility in two different views, and complement each other. In order to address the challenges from sparse and noisy data, we design two specific strategies to combine the two components. First, two components share the learned parameters or representations, so that the original representations of each component can be enhanced. More importantly, we adopt a co-teaching mechanism to reduce the influence of noise in training data. The core idea is to let the two components help each other by selecting more reliable training instances. The two strategies focus on representation enhancement and data enhancement, respectively. Compared with pure text-based matching models, the proposed approach is able to learn better data representations from limited or even sparse interaction data, which is more resistible to noise in training data. Experiment results have demonstrated that our model is able to outperform state-of-the-art methods for job-resume matching.

[![model details](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/model_pic.jpg)](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/model_pic.jpg)

## Datasets

We present the statistics of the linked dataset in the following table:

[![detail statistics](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/data_table.jpg)](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/data_table.jpg)

## Download and Usage

[Update] Data cannot be shared temporarily due to commercial reasons.

## References

If you use our dataset or useful in your research, please kindly cite our papers.

```
@inproceedings{bian2020learning,
  title={Learning to Match Jobs with Resumes from Sparse Interaction Data using Multi-View Co-Teaching Network},
  author={Shuqing Bian, Xu Chen, Wayne Xin Zhao, Kun Zhou, Yupeng Hou, Yang Song, Tao Zhang and Ji-Rong Wen},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={65--74},
  year={2020}
}
```

## Additional Notes

- The following people contributed to this work: Shuqing Bian, Xu Chen, Wayne Xin Zhao, Kun Zhou, Yupeng Hou, Yang Song, Tao Zhang and Ji-Rong Wen.
- If you have any questions or suggestions with this dataset, please kindly let us know. Our goal is to make the dataset reliable and useful for the community.
- For contact, send email to [bianshuqing@ruc.edu.cn](mailto:bianshuqing@ruc.edu.cn).
