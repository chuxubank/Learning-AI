# Logistic Regression as a Neural Network

## Binary Classification
In a binary classification problem, the result is a discrete value output.

## Logistic Regression
Logistic regression is a learning algorithm used in a supervised learning problem when the output $y$ are all either zero or one. The goal of logistic regression is to minimize the error between its predictions and training data.

## Logistic Regression Cost Function
To train the parameters $w$ and $b$, we need to define a cost function.

### Loss (error) function
The loss function measures the discrepancy between the prediction $\hat{y}^{(i)}$ and the desired output $y^{(i)}$.

In other words, the loss function computes the error for a single training example.

### Cost function
The cost function is the average of the loss function of the entire training set. We are going to find the parameters $w$ and $b$ that minimize the overall cost function.

## Gradient Descent
Want to find $w$, $b$ that minimize $J(w,b)$

## Computation graph
The computations of a neural network are organized in terms of a forward pass or a forward propagation step, in which we compute the output of the neural network, followed by a backward pass or back propagation step, which we use to compute gradients or compute derivatives. 

The computation graph explains why it is organized this way.