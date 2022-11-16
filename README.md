# nlp-project-vqa

## Introduction

Visual Question Answering models implemented in `MindSpore`. We re-implemented `baseline` model in paper [VQA: Visual Question Answering](https://arxiv.org/pdf/1505.00468.pdf), `stack attention` model in [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1511.02274.pdf), and `topdown attention` model in [Tips and Tricks for Visual Question Answering:
Learnings from the 2017 Challenge](https://openaccess.thecvf.com/content_cvpr_2018/papers/Teney_Tips_and_Tricks_CVPR_2018_paper.pdf), and made some improvements. It's our final group project for courses `Introduction to Natural Language Processing` at Zhejiang University.

- ###### What's VQA?

[Visual Question Answering (VQA)](https://visualqa.org/) is a type of tasks, where given an image and a question about the image, a model is expected to give a correct answer.

For example, a **visual** image looks like this:

![image](https://user-images.githubusercontent.com/58615742/202229329-e7a48b56-1d70-41bb-b460-40450a692ef4.png)

The **question** is: How many people are there in the picture?

The correct **answer** would be: "3"

- ###### What's MindSpore?

[MindSpore](mindspore.cn) is a new AI framework developed in Huawei.

## Challenges & Improvements

- ###### Processing of multi word answers

Considering some answers may contain multi words, such as `fly kite`, we do not make any word segmentation for the answer, but directly make the answer case coded as a unit, and the answer index is stored in a dictionary.

- ###### Handling of options

The json file provides the alternative answers. In the later stage of the experiment, we try to use this information to improve the accuracy.

10 alternative answers for a single `VQA` question is displayed below:


![image](https://user-images.githubusercontent.com/58615742/202229781-3bbc2e80-a3c7-434e-850f-bb65ef1d2d9b.png)

How to deal with alternative answers? Instead of simply ignoring them as previous proposed models in the aforementioned papers did, we adopt `Attention Mechanism` in our project. To be more specific, using input `images` and `questions` as `query`, `options_answer` as `key`:

$$
h = \tanh (W_q \cdot q + W_k \cdot k)
$$

$$
p = softmax(W_h \cdot h)
$$

$$
output = matMul(p, options)
$$

Through the `Attention Mechanism`, the model is able to focus on which of the alternative answers are more suitable for the correct answer according to the input questions and images. After multiple rounds of training, we can achieve better results.

## Results

- ###### Without options

|      | baseline | stack attention | topdown attention |
| ---- | -------- | --------------- | ----------------- |
| Acc  | 22.02%   | 27.63%          | 21.93%            |

- ###### Options

|      | baseline   | stack attention | topdown attention |
| ---- | ---------- | --------------- | ----------------- |
| Acc  | **41.43%** | **62.18%**      | **38.05%**        |

## Further Discussion

By observing the training results of aforementioned models, it is not difficult to find that the network structure with the attention mechanism can more fully extract the information of pictures and texts to achieve better results under `VQA` task.

Since MindSpore lacks some effective pre-training models, the `top-down attention` model we have implemented cannot further generate text and picture vectors, which hold fully extracted task information, by pre-training parameter matrix after obtaining fusion vectors.

In the light of this fact, we propose 36$\times$2048 dimensional feature maps corresponding to each image obtained through `fast R-CNN+visual genome+ResNet`, so as to obtain more accurate and rich image features at the beginning to achieve better classification effect.

The performance improvement achieved by introducing `options` method is very intuitive and natural: it is equivalent that the model has changed from completing the blank filling questions to completing the choices questions. At this point, the information of pictures and texts is integrated - that is, the model's understanding of the current problem can be used to select the most likely answer that the model thinks, using `attention mechanism` in the candidate set. 

## [Report in Chinese](https://github.com/HoffYoung/nlp-project-vqa/blob/main/report.pdf)

