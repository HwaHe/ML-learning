# Lec 3 Logistic Regression and Regulation

## 3.1 分类问题

在机器学习中，除了前面提到的回归问题，另一个很重要的是分类问题。在第三周，我们学习了利用Logistic Regression来进行二分类以及多分类，并且使用了正规化的方法来解决过拟合问题。

我们将因变量(**dependent variable**)可能属于的两个类分别称为负向类（**negative class**）和正向类（**positive class**），则因变量$y\in { 0,1 \\}$ ，其中 0 表示负向类，1 表示正向类。

![](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210304133506.png)

![](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210304133543.png)

可以看到，如果我们使用线性回归做分类问题，则决策边界（也就是图上的拟合直线）很容易受到一些“离群值”的影响。从而使得分类的准确度急剧下降。

而且，对于线性回归模型，输出值可能远大于1，也可能远小于0。使得整个模型看起来很奇怪。



## 3.2 模型假设

综合以上分析，我们得找到一个模型，拥有如下的性质：

1. 输出值在0~1之间。
2. 当x值很小或者很大的时候，模型基本进入一个平台。

综合考虑上面的两个性质，我们可以采用sigmoid函数：
$$
sigmoid(z) = \frac{1}{1+e^{-z}}
$$


其函数图像如下：

![](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210304135527.jpg)

那我们怎么做分类问题呢？可以简单处理一下：

当$sigmoid(z)>=0.5$时，预测 $y=1$。

当$sigmoid(z)<0.5$时，预测 $y=0$ 。

那我们如何处理多特征分类问题呢？可以和之前一样，使用线性组合的方式。因此有：
$$
z=\mathbf{\theta}^Tx
$$
因此，我们可以得到如下所示的模型假设：
$$
h_\theta\left(x^{(i)}\right)=sigmoid(\mathbf{\theta^T}\mathbf{x}), \\
where\quad sigmoid(z)=\frac{1}{1+e^{-z}}
$$
对模型的理解：

$h_\theta \left( x \right)$的作用是，对于给定的输入变量，根据选择的参数计算输出变量=1的可能性（**estimated probablity**）即$h_\theta \left( x \right)=P\left( y=1|x;\theta \right)$
例如，如果对于给定的$x$，通过已经确定的参数计算得出$h_\theta \left( x \right)=0.7$，则表示有70%的几率$y$为正向类，相应地$y$为负向类的几率为1-0.7=0.3。



## 3.3 代价函数

对于线性回归模型，我们定义的代价函数是所有模型误差的平方和。理论上来说，我们也可以对逻辑回归模型沿用这个定义，但是问题在于，当我们将${h_\theta}\left( x \right)=\frac{1}{1+{e^{-\theta^{T}x}}}$带入到这样定义了的代价函数中时，我们得到的代价函数将是一个非凸函数（**non-convex function**）。

![](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210304140841.jpg)

这意味着我们的代价函数有许多局部最小值，这将影响梯度下降算法寻找全局最小值。

线性回归的代价函数为：$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{1}{2}{{\left( {h_\theta}\left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}}$ 。
我们重新定义逻辑回归的代价函数为：$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{{Cost}\left( {h_\theta}\left( {x}^{\left( i \right)} \right),{y}^{\left( i \right)} \right)}$，其中

![](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210304140921.png)

${h_\theta}\left( x \right)$与 $Cost\left( {h_\theta}\left( x \right),y \right)$之间的关系如下图所示：

![](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210304141015.jpg)

这样构建的$Cost\left( {h_\theta}\left( x \right),y \right)$函数的特点是：当实际的  $y=1$ 且${h_\theta}\left( x \right)$也为 1 时误差为 0，当 $y=1$ 但${h_\theta}\left( x \right)$不为1时误差随着${h_\theta}\left( x \right)$变小而变大；当实际的 $y=0$ 且${h_\theta}\left( x \right)$也为 0 时代价为 0，当$y=0$ 但${h_\theta}\left( x \right)$不为 0时误差随着 ${h_\theta}\left( x \right)$的变大而变大。
将构建的 $Cost\left( {h_\theta}\left( x \right),y \right)$简化如下： 
$Cost\left( {h_\theta}\left( x \right),y \right)=-y\times log\left( {h_\theta}\left( x \right) \right)-(1-y)\times log\left( 1-{h_\theta}\left( x \right) \right)$
带入代价函数得到：
$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$
即：$J\left( \theta  \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$

此时，我们得到的代价函数$J(\theta)$是一个凸函数，没有局部最小值。

接下来我们就可以使用梯度下降算法来求解$\theta$了。

> **Repeat** {
> $\qquad \theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j} J(\theta)$
> (**simultaneously update all** )
> }


求导后得到：

> **Repeat** {
> $\qquad \theta_j := \theta_j - \alpha \frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {h_\theta}\left( \mathop{x}^{\left( i \right)} \right)-\mathop{y}^{\left( i \right)} \right)}}\mathop{x}_{j}^{(i)}$ 
> **(simultaneously update all** )
> }

接下来我们证明这个过程，为了简便表示，我们这里将$h_\theta\left(x^{(i)}\right)$简写为$h_\theta$，将$x^{(i)}$简写为$\mathbf x$，将$y^{(i)}$简写为$\mathbf y$，则有：
$$
\begin{align}
\frac{\partial}{\partial \theta_j}J(\theta)&=-\frac{1}{m} \cdot \bigg[\sum_{i=1}^m\big[\frac{y}{h_\theta}\cdot \frac{\partial h_{\theta}}{\partial \theta_j } + (y-1) \cdot \frac{1}{1-h_\theta} \cdot \frac{\partial h_\theta}{\partial \theta_j}\big]\bigg]\\
&=-\frac{1}{m} \cdot \bigg[\sum_{i=1}^m(\frac{y}{h_\theta} + \frac{y-1}{1-h_\theta})\cdot h_\theta^2 \cdot e^{-\theta^Tx^{i}}\cdot x_j^{i}\bigg]\\
&=\frac{1}{m}\sum_{i=1}^m(h_\theta - y)x_j^{i}
\end{align}
$$
代入即可得到$\theta$的更新公式。

注：虽然得到的梯度下降算法表面上看上去与线性回归的梯度下降算法一样，但是这里的${h_\theta}\left( x \right)=g\left( {\theta^T}X \right)$与线性回归中不同，所以实际上是不一样的。另外，在运行梯度下降算法之前，进行特征缩放依旧是非常必要的。



一些梯度下降算法之外的选择：
除了梯度下降算法以外，还有一些常被用来令代价函数最小的算法，这些算法更加复杂和优越，而且通常不需要人工选择学习率，通常比梯度下降算法要更加快速。这些算法有：**共轭梯度**（**Conjugate Gradient**），**局部优化法**(**Broyden fletcher goldfarb shann,BFGS**)和**有限内存局部优化法**(**LBFGS**) ，**fminunc**是 **matlab**和**octave** 中都带的一个最小值优化函数，使用时我们需要提供代价函数和每个参数的求导，下面是 **octave** 中使用 **fminunc** 函数的代码示例：

```octave
function [jVal, gradient] = costFunction(theta)

    jVal = [...code to compute J(theta)...];
    gradient = [...code to compute derivative of J(theta)...];
    
end
    
options = optimset('GradObj', 'on', 'MaxIter', '100');
    
initialTheta = zeros(2,1);
    
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

共轭梯度法，BFGS，L-BFGS需要有一种方法来计算$J(\theta)$，以及需要一种方法来计算其导数项，然后使用比梯度下降更复杂的算法来最小化代价函数。

这三种算法有许多有点：

比如，他们不需要我们来选择一个合适的学习率$\alpha$，其中使用线性搜索算法来自动尝试不同的学习速率$\alpha$，并且选择一个比较好的$\alpha$，而且他们往往比梯度下降算法收敛的快得多。



## 3.4 多类别分类

对于一个多类别分类问题，我们的数据集往往看起来是这个样子的：

![](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210304155943.png)

那我们应该怎么进行多类别分类呢？很简单，我们可以先遮住一部分绿色三角形样本，在正方形和叉之间应用二分类，然后同理迭代。这样，对于每个样本我们就会有三个预测，选择其概率最大的一个预测即可。



## 3.5 正规化

### 3.5.1 举例

下面是一个回归问题的例子：

![](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210304160225.jpg)

我们可以看到，当我们试图对一些样本加入很多多项式特征的时候，我们的模型在训练集上会表现的很好，却失去了泛化能力，这被成为“过拟合”；而当我们的模型的特征不足的时候，我们的模型在训练集上就会表现的很差，如第一个图般，这被成为“欠拟合”。自然而然的，我们就可以想到，我们可不可以加一个因子，来控制模型在过拟合和欠拟合上的表现，得到一个比较合适的模型。



### 3.5.2 代价函数

上面的回归问题中如果我们的模型是：
${h_\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}^2}+{\theta_{3}}{x_{3}^3}+{\theta_{4}}{x_{4}^4}$
我们可以从之前的事例中看出，正是那些高次项导致了过拟合的产生，所以如果我们能让这些高次项的系数接近于0的话，我们就能很好的拟合了。
所以我们要做的就是在一定程度上减小这些参数$\theta $ 的值，这就是正则化的基本方法。我们决定要减少${\theta_{3}}$和${\theta_{4}}$的大小，我们要做的便是修改代价函数，在其中${\theta_{3}}$和${\theta_{4}}$ 设置一点惩罚。这样做的话，我们在尝试最小化代价时也需要将这个惩罚纳入考虑中，并最终导致选择较小一些的${\theta_{3}}$和${\theta_{4}}$。
修改后的代价函数如下：$\underset{\theta }{\mathop{\min }}\,\frac{1}{2m}[\sum\limits_{i=1}^{m}{{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}^{2}}+1000\theta _{3}^{2}+10000\theta _{4}^{2}]}$

通过这样的代价函数选择出的${\theta_{3}}$和${\theta_{4}}$ 对预测结果的影响就比之前要小许多。假如我们有非常多的特征，我们并不知道其中哪些特征我们要惩罚，我们将对所有的特征进行惩罚，并且让代价函数最优化的软件来选择这些惩罚的程度。这样的结果是得到了一个较为简单的能防止过拟合问题的假设：$J\left( \theta  \right)=\frac{1}{2m}[\sum\limits_{i=1}^{m}{{{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta_{j}^{2}}]}$

其中$\lambda $又称为正则化参数（**Regularization Parameter**）。

<font color='red'> 注：根据惯例，我们不对${\theta_{0}}$ 进行惩罚。经过正则化处理的模型与原模型的可能对比如下图所示：</font>

![](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210304170906.jpg)

如果选择的正则化参数$\lambda$ 过大，则会把所有的参数都最小化了，导致模型变成 ${h_\theta}\left( x \right)={\theta_{0}}$，也就是上图中红色直线所示的情况，造成欠拟合。
那为什么增加的一项$\lambda =\sum\limits_{j=1}^{n}{\theta_j^{2}}$ 可以使$\theta $的值减小呢？
因为如果我们令 $\lambda$ 的值很大的话，为了使**Cost Function** 尽可能的小，所有的 $\theta $ 的值（不包括${\theta_{0}}$）都会在一定程度上减小。
但若$\lambda$ 的值太大了，那么$\theta $（不包括${\theta_{0}}$）都会趋近于0，这样我们所得到的只能是一条平行于$x$轴的直线。
所以对于正则化，我们要取一个合理的 $\lambda$ 的值，这样才能更好的应用正则化。
回顾一下代价函数，为了使用正则化，让我们把这些概念应用到到线性回归和逻辑回归中去，那么我们就可以让他们避免过度拟合了。



### 3.5.3 正规化线性回归

对于线性回归的求解，我们之前推导了两种学习算法：一种基于梯度下降，一种基于正规方程。

正则化线性回归的代价函数为：

$J\left( \theta  \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{[({{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta _{j}^{2}})]}$

如果我们要使用梯度下降法令这个代价函数最小化，因为我们未对$\theta_0$进行正则化，所以梯度下降算法将分两种情形：

$Repeat$  $until$  $convergence${

​                                                   ${\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}})$ 

​                                                   ${\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}+\frac{\lambda }{m}{\theta_j}]$ 

​                                                             $for$ $j=1,2,...n$

​                                                   }




对上面的算法中$ j=1,2,...,n$ 时的更新式子进行调整可得：

${\theta_j}:={\theta_j}(1-a\frac{\lambda }{m})-a\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}$ 
可以看出，正则化线性回归的梯度下降算法的变化在于，每次都在原有算法更新规则的基础上令$\theta $值减少了一个额外的值。

我们同样也可以利用正规方程来求解正则化线性回归模型，方法如下所示：

![](https://gallery-1259614029.cos.ap-chengdu.myqcloud.com/img/20210304171044.png)

图中的矩阵尺寸为 $(n+1)*(n+1)$。



### 3.5.4 正规化的逻辑回归模型

类似于上面的内容，我们也给逻辑回归加上一个正则化的表达式，得到代价函数：
$$
J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}+\frac{\lambda }{2m}\sum\limits_{j=1}^{n}{\theta _{j}^{2}}
$$
要最小化该代价函数，通过求导，得出梯度下降算法为：

$Repeat$  $until$  $convergence${

​                                                   ${\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}})$

​                                                  ${\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}+\frac{\lambda }{m}{\theta_j}]$

​                                                 $for$ $j=1,2,...n$

​                                                 }

注：看上去同线性回归一样，但是知道 ${h_\theta}\left( x \right)=g\left( {\theta^T}X \right)$，所以与线性回归不同。