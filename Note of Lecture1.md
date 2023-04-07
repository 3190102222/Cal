ML(Machine Learning)

# Curve Fitting
Curve fitting is the most basic problem.
Given $\{x_{i},y_{i}\}$, our aim is to find $f(x)$ s.t. $y_{i} \approx f(x_{i})$. We consider a parametric family $\{f_{\theta}(x):\theta \in \Theta\}$, $\theta$ is the parameter.

Regression
$f_{\theta}(x) = \omega ^T x +b,\theta = [\omega,b]$

Neural Network(NN)
$f_{\theta}(x) = \sum\limits_{ j = 1 }^{ m } \sigma(\omega_{j}^Tx+b_{j}),\theta =\{\omega_{i},b_{i}\}_{i=1}^m$, here $\sigma(x) = max\{x,0\}$

Transformers
Omit.

Define loss:
$L(\theta) = \sum\limits_{ j = 1 }^{ n }(y_{j}-f_{\theta}(x_{j}))^2$
We transform the above regression problem to optimization problem. We want to minimize $L(\theta)$, i.e. $\min\limits_{\theta}L(\theta)$. We have methods such as "Gradient Descent" of optimization.


## Learning function $\Rightarrow$ Learning distribution

Given $\{x_{i}\}_{i=1}^n$, we want to learn a distribution $P$.
Given $\{x_{i},y_{i}\}_{i=1}^n$, we want to learn a conditional distribution $P(Y|X)$. For example, $X$ refers to "text prompt" and $Y$ refers to "image".


## Maximum Likelihood Estimation(MLE)

1.Specify a parametric family $P_{\theta}(x)$, i.e. "density function".
2.Define likelihood
$$
L(\theta) = \dfrac{1}{n}\sum\limits_{ i = 1 }^{ n }\log P_{\theta}(X_{i}) = \mathbb{E}_{X \sim P_{data}}[\log P_{\theta}(X)]
$$
Here, $P_{data}$ is the empirical distribution of data $X_{i}$, namely that $P_{data}(x)=\sum\limits_{ i = 1 }^{ n }\delta(x-x_{i})/n$
3.Optimization:$\max\limits_{\theta}L(\theta)$, and $\hat\theta = \arg\max \{L(\theta):\theta \in \Theta\}$


## KL divergence

We have two distributioins $p,q$, and hope to define a "distance" $D(p,q)$ between $p,q$. Then minimize $D(P_{data},P_{\theta})$ i.e.
$$
  \min\limits_{\theta} D(P_{data},P_{\theta})  
$$

Define $KL(P_{data},P_{\theta}) = \mathbb{E}_{x \sim P_{data}}[\log P_{data}(x)-\log P_{\theta}(x)]$. It involves "Information Theory"

Strictly speaking, KL divergence is not a distance. It's positive definite but asymmetric.
1.$KL(P_{data},P_{\theta}) \ge 0$
2.$KL(P_{data},P_{\theta}) =0$ iff $P_{data} = P_{\theta}$
proof(why $KL(p,q) \ge 0$ ?)
$$
  KL(p,q) = \mathbb{E}_{x \sim p}[\log p(x)-\log q(x)] =-\mathbb{E}_{x \sim p}[\log \dfrac{p(x)}{q(x)}] = -\int p(x) \log \left( \dfrac{q(x)}{p(x)} \right) \, dx \ge -\log \left( \int p(x) \dfrac{q(x)}{p(x)} \, dx \right) = 0 
$$
The inequality is due to Jensen's Inequality.


To minimize KL divergence is equivalent to maximize the likelihood:
$$
  KL(P_{data},P_{\theta}) = \mathbb{E}_{P_{data}}[\log P_{data}(x)] - \mathbb{E}_{P_{data}}[\log P_{\theta}(x)]  
$$
Here, $\mathbb{E}_{P_{data}}[\log P_{data}(x)]$ is a constant. Hence, to minimize KL divergence is equivalent to maximize the likelihood.


Example: learning Gaussian distribution
$P_{\theta}(x) = \dfrac{1}{\sqrt{ 2\pi  }\sigma} \exp \left( -\dfrac{(x-\mu)^2}{2 \sigma^2}\right), \ \theta = [\mu,\sigma]$. Given $\{x_{i}\}^n_{i=1}$, 
$$
l(\theta) =\sum\limits_{ i = 1 }^{ n } \log P_{\theta}(x_{i}) =\sum\limits_{ i = 1 }^{ n } \left[ - \dfrac{(x-\mu)^2}{2\sigma^2} -\log \sigma -\log (\sqrt{ 2\pi }) \right]
$$
$$
  \overset{\text{take the derivative}}{\Rightarrow } \hat{\mu} =\dfrac{1}{n}\sum\limits_{ i = 1 }^{ n }X_{i}, \ \hat{\sigma}^2  =\dfrac{1}{n}\sum\limits_{ i = 1 }^{ n }(X_{i}-\hat{\mu})^2
$$


Generalized example:
Assume $P_{\theta}(y|x) = \dfrac{1}{\sqrt{ 2\pi  }\sigma_{\theta}(x)} \exp \left( -\dfrac{(y-\mu_{\theta}(x))^2}{2 \sigma_{\theta}(x)^2}\right)$, i.e. $y|x \sim N(\mu_{\theta}(x),\sigma_{\theta}(x)^2)$. Here, the mean value and the variance is a function with parameter $\theta$.
$$
    l(\theta) =\sum\limits_{ i = 1 }^{ n } \log P_{\theta}(y_{i}|x_{i}) = \dots
$$

Example: discrete distribution
Given $\{X_{i}\}_{i=1}^n$ independent and identically distributed, $X_{i} \in \{1,2,\dots ,K\}$, and $P(X_{i} = j) = p_{j}$, for $j=1,2,\dots,K$. We have $\sum\limits_{ j }p_{j} =1, \ p_{j} \ge0$. Write $p_{i}=\dfrac{\exp(\theta_{i})}{\sum\limits_{ j = 1 }^{ K } \exp(\theta_{j})}, \ \theta=[\theta_{1},\theta_{2},\dots,\theta_{K}]$
$$
  l(\theta) =\sum\limits_{ i = 1 }^{ n }\log P_{\theta}(X_{i})=\sum\limits_{ i = 1 }^{ n }\log \left(\dfrac{\exp(\theta_{X_{i}})}{\sum\limits_{ j = 1 }^{ K }\exp(\theta_{X_{j}})} \right)  
$$
$$
  \hat{\theta} =\arg \max\limits_{\theta} L(\theta) \Rightarrow P_{\hat{\theta}}(k) = \dfrac{\#\{X_{i} =k\}}{n}  
$$

Generalized example: Cross entropy loss
Given $\{x_{i},y_{i}\}_{i=1}^n$, $y_{i} \in \{1,2,\dots,K\}$. $P_{\theta}(y=k|x) = \dfrac{\exp(f_{\theta_{k}}(x))}{\sum\limits_{ j = 1 }^{ K }\exp(f_{\theta_{j}}(x))}$
$$
  l(\theta) = \sum\limits_{ i = 1 }^{ n }\log P(y_{i}|x_{i}) = \dots   
$$


An intuitive insight into the motivation of KL divergence
Define "entropy" $H(p) =-\mathbb{E}_{x \sim p}[\log p(x)]$.
Define "cross entropy" $CH(p,q) =-\mathbb{E}_{x\sim P}[\log q(x)] =-\int p(x)\log q(x) \, dx$
claim. $\max\limits_{q}CH(p,q) =H(p)$, i.e. taking $q=p$ maximizes $CH(p,q)$
Then, define $KL(p,q)=H_{p} -CH(p,q) =\max\limits_{q'}CH(p,q') -CH(p,q)$ which characterizes the gap between $CH(p,q)$ and the maximum. 



## How to deal with high dimensional data?

Notations: $\{X^{(i)}\}_{i=1}^n$, $X^{(i)} =[X^{(i)}_{1},X^{(i)}_{2},\dots,X^{(i)}_{d}] \in \mathbb{R}^d$


## Auto-regressive Model

$$
  P(X_{1},X_{2},\dots,X_{d}) = P(X_{1})P(X_{2}|X_{1})P(X_{3}|X_{1},X_{2})\dots = \prod_{i=1}^{d}P(X_{i}|X_{pa_{i}}), \ pa_{i}=\{1,2,\dots,i-1\}  
$$
Then,   $\max\limits_{\theta}\sum\limits_{ i = 1 }\log P_{\theta}(X^{(i)}_{1},X^{(i)}_{2},\dots,X^{(i)}_{d})$

Our idea is "divide and conquer".

Example: Digital Regression
Given $\{x^{(i)},y^{(i)}\}^n_{i=1}, \ y^{(i)} \in \mathbb{R}$, we want to compute $P(y|x)$. Write $y = \sum\limits_{ i = -m }^{ m }y_{i}10^i$, here $y_{i}$ is the i-th digit, $y_{i} \in \{0,1,\dots,9\}$. We use auto-regressive method to learn each digit.
$$
  P(y|x) =P(y_{m}|x)P(y_{m-1}|x,y_{m})P(y_{m-2}|y_{m-1},y_{m},x)  
$$
Then we can use "cross entropy loss" method to learn $P(y_{m}|x)$ .