# nlp-project-vqa

## Introduction

Visual Question Answering models implemented in `MindSpore`. We re-implemented `baseline` model in paper [VQA: Visual Question Answering](), `stack attention` model in [Stacked Attention Networks for Image Question Answering](), and `topdown attention` model in [Tips and Tricks for Visual Question Answering:
Learnings from the 2017 Challenge]() as baselines, and made some improvements. It's our final group project for courses `Introduction to Natural Language Processing` at Zhejiang University.

- ###### What's VQA?

**Visual Question Answering (VQA)** is a type of tasks, where given an image and a question about the image, a model is expected to give a correct answer.

For example, a **visual** image looks like this:

![image](https://user-images.githubusercontent.com/58615742/202229329-e7a48b56-1d70-41bb-b460-40450a692ef4.png)

The **question** is: How many people are there in the picture?

The correct **answer** would be: "3"

- What's MindSpore?

[MindSpore](mindspore.cn) is a new AI framework developed in Huawei.

## Challenges & Improvements

- ###### Processing of multi word answers

Considering some answers do not contain unique words, such as `fly kite`, we do not make any participle for the answer, but directly answer

Case is coded as a unit, and the answer index is stored in a dictionary.

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

## Report in Chinese

