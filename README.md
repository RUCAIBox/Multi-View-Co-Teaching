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

[![detail statistics](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/model_pic.jpg)](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/model_pic.jpg)

## Datasets

We present the statistics of the linked dataset in the following table:

[![detail statistics](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/data_table.jpg)](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/data_table.jpg)

## Download and Usage

### By using the datasets, you must agree to be bound by the terms of the following [licence](https://github.com/RUCAIBox/Multi-View-Co-Teaching/blob/master/README.md#Licence).

By using the datasets, you must agree to be bound by the terms of the following license.

Then mail to [bianshuqing@ruc.edu.cn] and cc Wayne Xin Zhao via [batmanfly@qq.com] and Yang Song via [songyang@kanzhun.com] and your supervisor, and copy the license in the email. We will send you the datasets by e-mail when approved.


## Licence

By using the datasets, you must agree to be bound by the terms of the following licence.

```
Licence agreement
This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the dataset comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we do not accept any responsibility for errors or omissions. 
2. That you include a reference to the ”BOSS Zhipin“ dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our References; for other media cite our preferred publication as listed on our website or link to the dataset website.
3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
5. That all rights not expressly granted to you are reserved by us (Wayne Xin Zhao, School of Information, Renmin University of China & Yang Song, BOSS Zhipin).
```

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
