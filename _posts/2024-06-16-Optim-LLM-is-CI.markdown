---
layout: post
title:  "Optimizing Language Models for Human Preferences is a Causal Inference Problem"
date:   2024-06-16 11:00:00 +0800
categories: paper_reading
---

## Backgrounds

The demerits of DPO: 
1. Need a reference model: resource consumption is **doubled**.
2. Preference dataset needs **annotation** by human or LLM.
3. The objective of DPO **does not necessarily align with** preference alignment.
> The below term only constraint the objective **in a probabilistic way**.
> 
> $$
> P^\*_{BT}(y_1\succ y_2|x)=\frac{\exp(r^\*(x,y_1))}{\exp(r^\*(x,y_1))+\exp(r^\*(x,y_2))}.
> $$

**Recall**: Why preference optimization is needed?
* Learning an optimal language model can be difficult due to **the presence of unmeasured confounding in the training data**: external factors that affect both readers’ choice of texts to read and how they tend to respond to those texts.

The solution to this optimization problem finds **how to intervene on the text distribution of the generating model to best cause an optimal outcome**.
Could we use the abundant **Direct Outcome Dataset**?

## Utilize Direct Outcome Dataset

General Objective on direct outcome dataset: optimize the average outcome among individuals who observed the text $X$ and can be learned from $D_O$.

$$
\arg\max_{f}\mathbb{E}_{X\sim P^f}[\mathbb{E}_{D_O}[Y|X]].
$$

However, this objective does not necessarily align with the true optimization goal due to **selection bias**.
We define $g(x)\equiv\mathbb{E}_{Y(\cdot)\sim \mathcal{G}}[Y(x)]$ as the average outcome if all individuals in the population were given text $x$.
The goal is to maximize the value function:

$$
V(f)\equiv\mathbb{E}_{X\sim P^f}[g(X)].
$$

However, not all individuals in the population can be accessed. So we need to estimate this.
we can link the value function to **the randomization dataset $D^R$**:

$$
V_{IPW}(f)=\mathbb{E}\_{P^R}\big\[\mathbb{E}\_{P_y^R}[\frac{P^f(X)}{P^R(X)}\cdot Y]\big\]
$$

Then, we can estimate it by $\hat{V_{IPW}}$.
If $D^R$ is randomization dataset, we can show that $\hat{V_{IPW}}$ has no bias. 

$$
\hat{V_{IPW}}(f)=\frac{1}{n}\sum_{i=1}^{n}\frac{P^f(X_i)}{P^R(X_i)}Y_i
$$

However, IPW has HIGH variance.
Apart from Direct Outcome Datasets, we also have lots of unlabeled texts.
So we can predict outcomes on unlabeled texts, i.e. **IPW + Outcome Modeling = Doubly Robust Estimator**:

$$
V_{DR}(f)=\mathbb{E}\_{P^R}\big\[\mathbb{E}\_{P_y^R}[\frac{P^f(X)}{P^R(X)}\cdot Y - g(X)]\big\] + \mathbb{E}_{X\sim P^f}[g(X)].
$$

Consider the outcome modeling term $\mathbb{E}_{X\sim P^f}[g(X)]$ first.
It is difficult to optimize $g$ as $f$ is also updated.
To remedy this, we can fix the language model:

$$
\mathbb{E}\_{X\sim P^f}[g(X)]=\mathbb{E}\_{X\sim P^{f^0}}\big\[\frac{P^f(X)}{P^{f^0}(X)}g(X)\big\].
$$

We can create a Monte Carlo estimate of this by drawing texts $\tilde{X_1},\cdots,\tilde{X_m}\sim P^{f^0}$ and computing:

$$
\hat{V_{out}}(f)=\frac{1}{m}\sum_{j=1}^m\frac{P^f(\tilde{X}_j)}{P^{f^0}(\tilde{X}_j)}\hat{g}(\tilde{X}_j),
$$

where $\hat{g}$ is a model trained to predict $Y$ from $X$.

Finally, the doubly robust value function $V_{DR}$ can be estimated as the combination of the above 2 terms:

$$
\hat{V}\_{DR}(f)=\frac{1}{n}\sum\_{i=1}^n\frac{P^f(X_i)}{\hat{P^R}(X_i)}\cdot(Y_i - \hat{g}(X_i))+\frac{1}{m}\sum_{j=1}^m\frac{P^f(\tilde{X}_j)}{P^{f^0}(\tilde{X}_j)}\hat{g}(\tilde{X}_j).
$$

$\hat{V}\_{DR}$ has no bias if either of the two terms hold:
1. $\hat{P^R}(X)=P^R(X)$
2. $\hat{g}(X)=g(X)$