# Lec 1. Introduction

## 1.1 "Batch" Gradient descent algorithm

**Cost function here**: $J(\theta_0, \theta_1)$

![image-20210225102242953](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210225102243.png)

我们先看一元线性回归的批量梯度下降算法的迭代怎么推导：
$$
\begin{align}
\textbf{Cost function: } h_\theta\big(x^{(i)}\big) &= \theta_0 + \theta_1 x\\
\frac{\partial}{\partial \theta_j}J(\theta_0, \theta_1) &= \frac{\partial}{\partial \theta_j} \cdot \frac{1}{2m} \cdot \sum_{i=1}^m\bigg(h_\theta({x^{(i)}}) - y^{i}\bigg)^2\\
&=\frac{\partial}{\partial \theta_j} \cdot \frac{1}{2m} \cdot \sum_{i=1}^m
(\theta_0 + \theta_1x^{(i)} - y^i)^2 \\
\\
j=0: \frac{\partial}{\partial \theta_0}J(\theta_0, \theta_1) &= \frac{1}{m} \cdot \sum_{i=1}^m\bigg(h_\theta\big({x(i)}\big) - y^{i}\bigg)\\
j=1: \frac{\partial}{\partial \theta_0}J(\theta_0, \theta_1) &= \frac{1}{m} \cdot \sum_{i=1}^m\bigg(h_\theta\big({x(i)}\big) - y^{i}\bigg)\cdot x^{(i)}\\
\end{align}
$$
以上是如何推导出来的呢？我们可以看如下的推导：

<img src="https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210225113505.jpg"/>

