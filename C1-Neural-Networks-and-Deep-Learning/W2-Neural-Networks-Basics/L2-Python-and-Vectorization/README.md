# Python and Vectorization
## Vectorization
Vectorization is basically the art of getting rid of explicit folders in your code.

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
z = np.dot(w, x) + b
```
This `numpy` function take the advantage of SIMD(Single Instruction Mutiple Data) in CPU and GPU

:::

[Demo](https://nbviewer.jupyter.org/github/chuxubank/Learning-AI/blob/master/C1-Neural-Networks-and-Deep-Learning/W2-Neural-Networks-Basics/L2-Python-and-Vectorization/Vectorization%20demo.ipynb)
