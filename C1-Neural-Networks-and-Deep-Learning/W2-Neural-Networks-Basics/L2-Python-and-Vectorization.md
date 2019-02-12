# Python and Vectorization
## Vectorization
Vectorization is basically the art of getting rid of explicit folders in your code.

When compute $z=w^Tx+b$, where $w=[\vdots], x=[\vdots]; w,x \in \mathbb{R}^{n_x}$

::: warning Non-vectorized
```python
z = 0
for i in range(n_x):
	z += w[i] * x[i]
z += b
```
:::

::: tip Vectorized
```python
import numpy as np
z = np.dot(w, x) + b
```
This `numpy` built-in function take much better advantage of parallelism to do your computations much faster.

Both GPU and CPU have parallelization instructions, aka SIMD (Single Instruction Multiple data).

:::

[Demo](https://nbviewer.jupyter.org/github/chuxubank/Learning-AI/blob/master/C1-Neural-Networks-and-Deep-Learning/W2-Neural-Networks-Basics/L2-Python-and-Vectorization/Vectorization%20demo.ipynb)


## More Vectorization Examples
### Neural network programming guideline
*Whenever possible, avoid explicit for-loops.*

If ever you want to compute $u=Av$

::: warning Non-vectorized

```python
u = np.zero(n,1)
for i ...
	for j ...
		u[i] += A[i][j] * v[j]
```
:::

::: tip Vectorized

```python
u = np.dot(A,v)
```
:::

### Vectors and matrix valued functions
Say you need to *apply the exponential operation on every element of a matrix/vector*.

$v = \begin{bmatrix} v_1 \\ \vdots \\ v_n \end{bmatrix}, u = \begin{bmatrix} e^{v_1} \\ \vdots \\ e^{v_n} \end{bmatrix}$

::: warning Non-vectorized

```python
u = np.zeros((n, 1))
for i in range(n):
	u[i] = math.exp(v[i])
```
:::

::: tip Vectorized

```python
u = np.exp(v)
```
:::

```python
# other vector value functions in numpy
np.log(v)
np.abs(v)
np.maximum(v, 0)
v**2	# takes the element-wise square of each element of v.
1/v 	# takes the element-wise inverse
```
### Logistic regression derivatives
::: warning Non-vectorized
```python
# Pseudocode
J = 0, dw1 = 0, dw2 = 0, db = 0
for i = 1 to m:
	z[i] = w_T * x[i] + b
	a[i] = sigma(z[i])
	J += -(y[i] * log(y_hat[i]) + (1 - y_i) * log(1 - y_hat[i]))
	dz[i] = a[i] * (1 - a[i])
	dw_1 += x_1[i] * dz[i]
	dw_2 += x_2[i] * dz[i]
	db += dz[i]
J /= m, dw_1 /= m, dw_2 /= m, db /= m

w_1 = w1 - alpha * dw_1
w_1 = w1 - alpha * dw_1
b = b - alpha * db
```
:::

::: tip Vectorized
```python {2,8,10}
# Pseudocode
J = 0, dw = np.zeros(n_x, 1), db = 0
for i = 1 to m:
	z[i] = w_T * x[i] + b
	a[i] = sigma(z[i])
	J += -(y[i] * log(y_hat[i]) + (1 - y_i) * log(1 - y_hat[i]))
	dz[i] = a[i] * (1 - a[i])
	dw += x[i] * dz[i]
	db += dz[i]
J /= m, dw /= m, db /= m

w_1 = w1 - alpha * dw_1
w_1 = w1 - alpha * dw_1
b = b - alpha * db
```
:::