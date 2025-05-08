---
title: "Branch Train Stack (BTS) For the Aurora-M2 Pretrained Models"
slug: bts-aurora-m2
description: "We introduce a novel Phased Training approach called Branch-Train-Stack that is highly efficient in terms of compute requirements while offering a simple process for debugging and data preparation."
category:
  - One
tags:
  - Training
  - Efficiency
pubDate: 2025-04-28
cover: https://raw.githubusercontent.com/aurora-lm/aurora-lm.github.io/main/assets/images/bts/bts-logo.png
coverAlt: Branch Train Stack Diagram
author: Aurora-M2 Team
---
**By: [Huu Nguyen](https://www.linkedin.com/in/huu-ai-machine-learning/), [Harsh Raj](hraj172.github.io), [Ken Tsui](https://www.linkedin.com/in/ken-tsui-06889b29/?originalSubdomain=uk), [Minh Chien Vu](https://scholar.google.com/citations?user=wcbZoCgAAAAJ&hl=en), [Felix Friedrich](https://www.ml.informatik.tu-darmstadt.de/people/ffriedrich/index.html), [Sonny Vu](https://scholar.google.com/citations?user=kFY-kEUAAAAJ&hl=en), [Diganta Misra](https://digantamisra98.github.io/), [Marianna Nezhurina](https://scholar.google.ru/citations?user=2KPv4VYAAAAJ&hl=en), [Victor May](https://mrcabbage972.github.io/) -- Apr 28, 2025**

<div align="center">

<strong>Huggingface:</strong> <a href="https://huggingface.co/ontocord" style="color: #1f6feb;" target="_blank"><strong>Aurora-M2</strong></a>

</div>

## Introduction

For training Aurora-M2, we introduce a novel Phased Training approach that is highly efficient in terms of compute requirements while offering a simple process for debugging and data preparation throughout the training process.

## Phased Training

We are adopting a phased approach to training. Our strategy for AuroraM-2 aims to achieve multiple objectives simultaneously: high data quality, expert specialization, and scalable performance across various parameter regimes (3B, 8B, and 20B), all while using significantly less compute than conventional large-scale models.​

We propose a novel training scheme optimized for low-resource environments and infrastructures with limited inter-node connectivity. As the proverb goes, necessity is the mother of invention; we devised this process to address our compute constraints. This approach, which we call Branch-Train-Stack (BTS), is inspired by prior [work](https://arxiv.org/abs/2208.03306) from Meta AI but introduces key innovations tailored for our context.

## The Branch-Train-Stack Process

The process can be broken down into several key stages:

### Initialization

In our [previous blog](https://aurora-lm.github.io/posts/mixturevitae) we collected the dataset, MixtureVitae: A Permissive, High-Performance, Open-Access Pretraining Dataset. We sample chunks from this dataset across our training process.

Our process begins with training an initial seed model—a 3-billion-parameter model initialized from scratch based on the *Qwen2.5-14B* architecture, with decreased number of layers to produce a 3B model. We sample roughly 5 billion heterogeneous tokens from MixtureVitae to serve as the base dataset for the seed model. Early experiments involve training several mixtures derived from MixtureVitae and evaluating them to select the best-performing model. For initial validation, we utilize 5 billion tokens for training at each stage. However, the training pipeline has been scaled to accommodate 20 billion tokens per stage per expert.

<figure>
  <img src="https://raw.githubusercontent.com/aurora-lm/aurora-lm.github.io/main/assets/images/bts/bts-flow.png" alt="Branch Train Stack Process">
  <figcaption>The Branch-Train-Stack (BTS) process.</figcaption>
</figure>

Inspired by prior [works](https://arxiv.org/abs/2302.08582) and the use of instructions in pretraining as in [this work](https://arxiv.org/abs/2211.09085) and our prior Aurora-M1 models, we perform alignment during pretraining to instill desired behaviors early, rather than only addressing misalignments during post-training. We observet that simply injecting refusals early in the training process will create undesirable behaviours, so we have instead created pro-social reasoning traces to inject alignment instructions. Specifically, we use an in-house data synthesis pipeline to create synthetic instructions that incorporate EU AI Act policies into the model. Our targeted data generation pipeline allows us to adjust the data mix to focus on areas where the student model underperforms, thereby improving performance. This novel data synthesis pipeline generates data from scratch in a fully controlled way; we discuss it in more detail in our [AutoRedTeam blog](https://aurora-lm.github.io/posts/autoredteam/).


### Branch-Train-Stack

After training the seed model, we branch the training into 8 specialized expert models, each initialized from the base model. For our final training experiment, we plan to use 20 experts, with curated data spanning 20 diverse categories. These models are trained separately on domain-specific datasets (e.g., business, fiction, math, medical/health), with a data composition of 50% heterogeneous data and 50% expert-specific data. 

Within the heterogeneous portion, we intentionally repeat some data from earlier training stages to minimize unexpected distribution drift. While these numbers were chosen for the initial test, we are still experimenting with the ratio and repetition rate for training. The training is conducted independently for each expert model, ensuring specialization without intercommunication between branches. This approach significantly reduces compute requirements and the need for high-speed inter-node communication between large node clusters.

After training, all eight models are merged to form a new base model. Inspired by state-of-the-art work, we employed the [DARE-TIES](https://arxiv.org/abs/2306.01708) merging algorithm with a density of 0.9 and assigned a weight of 0.05 to each merged model. While we initially chose equal weighting for each expert, we plan to conduct further ablation studies to optimize these parameters. This iterative process is repeated, progressively refining the model by integrating both general and specialized knowledge over multiple training cycles.

When scaling to larger models, we adopt a progressive stacking approach. After the first two iterations—training on approximately 8×5 + 5 = 45 billion tokens—we perform the first stacking, creating an 8-billion-parameter model. This stacking process is derived from previous [works](https://arxiv.org/abs/2405.15319) on model stacking. In our final training phase, we plan to train each expert on 20 billion tokens during each stage. We will stack the 3-billion-parameter model after 9 iterations of BTS to create an 8-billion-parameter model, and then stack again after 5 more iterations to create a 20-billion-parameter model, and then train for one more iteration.

## Preliminary Results

We conducted a series of ablation studies to assess the viability of our proposed training scheme. **For the preliminary experiments and validation, we only used 5B training tokens at each stage**. Table 2 presents the results from the initial phase and the expert phase up to two iterations, including the outcomes of the stacking model after these iterations. All evaluations were performed using Hugging Face's Lighteval framework, with the exception of HumanEval, which was done using BigCodeBench.

| Stage | Expert | HumanEval (pass@100) | GSM8K (lighteval) | GSM8k (lm_eval) | ARC Challenge | Winogrande | MMLU | Hellaswag |
|-------|--------|----------------------|-------------------|-----------------|---------------|------------|------|-----------|
| 0 | init | 0 | 0 | 0.0235 | 0.2448 | 0.5067 | 0.2543 | 0.2966 |
| 1 | wiki | 0 | 0 | 0.0243 | 0.2448 | 0.5082 | 0.2542 | 0.2964 |
| 1 | formatted_text | 0 | 0 | 0.0288 | 0.2474 | 0.5161 | 0.2499 | 0.3123 |
| 1 | how_to | 0.0223 | 0 | 0.0152 | 0.2457 | 0.4932 | 0.2468 | 0.3342 |
| 1 | law | 0.0219 | 0 | 0.0182 | 0.2542 | 0.4988 | 0.2556 | 0.3105 |
| 1 | news | 0.0304 | 0.0015 | 0.0121 | 0.2482 | 0.5051 | 0.2545 | 0.3156 |
| 1 | software | 0.0162 | 0 | 0.0212 | 0.2372 | 0.5177 | 0.2524 | 0.3068 |
| 1 | fictional_lyrical | 0.0115 | 0 | 0.0182 | 0.2525 | 0.5114 | 0.2478 | 0.3147 |
| 1 | math | 0.0805 | 0.0007 | 0.0356 | 0.2602 | 0.5098 | 0.2587 | 0.3154 |
| 1 | merged | 0.0558 | 0.0015 | 0.0182 | - | - | - | - |
| 2 | fictional_lyrical | 0 | 0 | 0.0174 | 0.2576 | 0.5177 | 0.2446 | 0.324 |
| 2 | math | 0.0758 | 0 | 0.0318 | 0.25 | 0.509 | 0.2546 | 0.3143 |
| 3 | math_stacked | 0 | - | - | 0.2542 | 0.5059 | 0.2531 | 0.3202 |

*Table 2: Preliminary results of BTS training across different phases and experts.*

For evaluation, we selected tasks commonly used for assessing small language models, such as those employed in SmolLM evaluations. These tasks are relatively less complex than current standards, providing early indicators of improvement during initial training stages. At this early stage, we do not anticipate the model to have acquired extensive world knowledge or factual information, as such competencies typically require more extensive training. 

However, we expect evaluation results for math and code tasks to show an increasing trend, as these are relatively logic-oriented tasks rather than reliant on memorization. This expectation aligns with our observed results. The results indicate that our pipeline is effective, showing improvements in model scores, particularly for logic, code, and math-related tasks—areas often considered primary indicators of learning progress.

## Conclusion

The Branch-Train-Stack (BTS) approach offers a promising and efficient method for training large language models with limited computational resources. By leveraging specialized expert models and progressive scaling, we can build powerful models that perform well across various parameter regimes while maintaining high quality and efficiency.

Our preliminary results demonstrate the effectiveness of this approach, and we continue to refine and optimize our training process. Stay tuned for more updates as we progress through our training phases and reach our final 20B model.

Looking forward, we foresee more optimization opportunities in improving the serving performance. For instance, [DeltaMoE](<https://openreview.net/forum?id=FJ7Z8H6elV&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FWorkshop%2FSLLM%2FAuthors%23your-submissions>) proposes compressing the deltas in each expert, leading to improved efficiency and performance. We believe that BTS has the potential to deliver benefits not only in training but also in the efficient deployment and serving of large models.

## Citation
```bibtex
@misc{bts_aurora_2025,
  author       = {Huu Nguyen, Harsh Raj, Ken Tsui, Vu Minh Chien, Felix Friedrich, Diganta Misra, Victor May, Marianna Nezhurina},
  title        = {Branch Train Stack (BTS) For the Aurora-M2 Pretrained Models},
  howpublished = {https://aurora-lm.github.io/posts/bts-aurora-m2},
  note         = {Accessed: 2025-04-28},
  year         = {2025}
}
```
