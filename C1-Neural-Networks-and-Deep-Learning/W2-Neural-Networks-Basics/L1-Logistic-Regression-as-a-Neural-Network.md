# Logistic Regression as a Neural Network

## Binary Classification
In a binary classification problem, the result is a discrete value output.

## Logistic Regression
Logistic regression is a learning algorithm used in a supervised learning problem when the output $y$ are all either zero or one. The goal of logistic regression is to minimize the error between its predictions and training data.

### Cat vs No-cat
Given an image represented by a feature vector 𝑥, the algorithm will evaluate the probability of a catbeing in that image.

$$
\text { Given } x , \hat { y } = P ( y = 1 | x ) , \text { where } 0 \leq \hat { y } \leq 1
$$
The parameters used in Logistic regression are:
- The input features vector: $x \in \mathbb { R } ^ { n _ { x } }$, where $n _ { x }$ is the number of features
- The training label: $y \in 0,1$
- The weights: $w \in \mathbb { R } ^ { n _ { x } }$, where $n _ { x }$ is the number of features
- The threshold: $b \in \mathbb { R }$
- The output: $\hat { y } = \sigma \left( w ^ { T } x + b \right)$
- Sigmoid function: $s = \sigma \left( w ^ { T } x + b \right) = \sigma ( z ) = \frac { 1 } { 1 + e ^ { - z } }$

$\left( w ^ { T } x + b \right)$ is a linear function $( a x + b )$, but since we are looking for a probability constraint between $[ 0,1 ] ,$ the sigmoid function is used.

## Logistic Regression Cost Function
To train the parameters $w$ and $b$, we need to define a cost function.

### Recap
$\hat { y } ^ { ( i ) } = \sigma \left( w ^ { T } x ^ { ( i ) } + b \right) ,$ where $\sigma \left( z ^ { ( i ) } \right) = \frac { 1 } { 1 + e ^ { - z ^ { ( i ) } } }$

Given $\left\{ \left( x ^ { ( 1 ) } , y ^ { ( 1 ) } \right) , \cdots , \left( x ^ { ( m ) } , y ^ { ( m ) } \right) \right\} ,$ we want $\hat { y } ^ { ( i ) } \approx y ^ { ( i ) }$

::: tip
$x ^ { ( i ) }$ : the i-th training example
:::

### Loss (error) function
The loss function measures the discrepancy between the prediction $\hat{y}^{(i)}$ and the desired output $y^{(i)}$.

In other words, the loss function computes the error for a single training example.

$L \left( \hat { y } ^ { ( i ) } , y ^ { ( i ) } \right) = \frac { 1 } { 2 } \left( \hat { y } ^ { ( i ) } - y ^ { ( i ) } \right) ^ { 2 }$
  - We do not use this because of multiple minimums.

$L \left( \hat { y } ^ { ( i ) } , y ^ { ( i ) } \right) = - \left( y ^ { ( i ) } \log \left( \hat { y } ^ { ( i ) } \right) + \left( 1 - y ^ { ( i ) } \right) \log \left( 1 - \hat { y } ^ { ( i ) } \right) \right)$
  - If $y ^ { ( i ) } = 1 : L \left( \hat { y } ^ { ( i ) } , y ^ { ( i ) } \right) = - \log \left( \hat { y } ^ { ( i ) } \right)$

### Cost function
The cost function is the average of the loss function of the entire training set. We are going to find the parameters $w$ and $b$ that minimize the overall cost function.

## Gradient Descent
Want to find $w$, $b$ that minimize $J(w,b)$

## Computation graph
The computations of a neural network are organized in terms of a forward pass or a forward propagation step, in which we compute the output of the neural network, followed by a backward pass or back propagation step, which we use to compute gradients or compute derivatives. 

The computation graph explains why it is organized this way.

## Derivatives with a Computation Graph
It is used to do derivative calculations for the function $J$.
In calculus, it is called the `chain rule`.
So indeed, terminology of backpropagation if you want to compute the derivative of this final output variable with respect to v, then we've done one step of `backpropagation`.

## Logistic Regression Gradient Descent
Using the computation graph is a little bit of an overkill for deriving gradient descent for logistic regression.