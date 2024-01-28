---
typora-copy-images-to: README.assets
---







$$
y_i(t) = w_{i,1}x(t) + w_{i,0} + \varepsilon_i(t) \quad \text{for } w_{i,2}x(t) + 1
$$


$$
y_i(t) = 
\frac{w_{i,1}^\mathsf{T} x(t) + w_{i,0} + \varepsilon_i(t)}{w_{i,2}^\mathsf{T}x(t) + 1}
$$

$$
\text{where } x(t) \in \mathbb{R}^d \text{ is the list of all measurements at time } t, \varepsilon_i(t) \text{ is an i.i.d. noise, and } (w_{i,0}, w_{i,1}, w_{i,2}) \in \mathbb{R} \times \mathbb{R}^d \times \mathbb{R}^d \text{ are the parameters of the model.}
$$



$$
y(t) = \frac{w_{1}^\mathsf{T} \tilde{x}(t) + w_{0}}{w_{2}^\mathsf{T}\tilde{x}(t) + 1}
$$




In training set, if we minimize the $\frac{1}{2} \|Aw - b\|$ by finding a $w'$ that let $Aw' = b$, which means for every sample in training set(for every $t$), we have $A_t w' = b_t = y(t)$. 

So,  $\forall t$ in training set,  $y(t) = \frac{w_{1}^\mathsf{T} \tilde{x}(t) + w_{0}}{w_{2}^\mathsf{T}\tilde{x}(t) + 1}$ 





In order to improve the generalization power of the model, we consider a $l_2$ regularization :
$$
\min_w \frac{1}{2} \|Aw - b\|_2^2 + \frac{\lambda}{2} \|w\|^2
$$
where $\lambda = 100$. Solve this problem and compare the test mean square error with the unregularized one.





---

---


$$
x = \arg\min_x g(x) + h(x)
$$

- $g(x)$ is differentiable convex function
- $h(x)$ is non-differentiable convex function
  - $h(x) = 0$, this degrades to gradient descent algorithm
  - $h(x) = I_C$, Projected gradient descent
  - $h(x) = \lambda \|x\|_1$ , Iterative Shrinkage-Thresholding Algorithm, ISTA



**LASSO :**
$$
\begin{align*}
x &= \arg\min_x g(x) + h(x) \\

x &= \arg \min_x \underbrace{\overbrace{\frac{1}{2}\| Ax - b \|^2_2}^{g(x)} + \overbrace{\lambda \|x\|_1}^{h(x)}}_{f(x)} \\

f'(x) &= A^{\mathsf{T}}(Ax - b) + \lambda \text{sgn}(x) \\


\end{align*}
$$

$$
\begin{align*}
x &= \arg\min_x g(x) + h(x) \\

x &= \arg \min_x \underbrace{\overbrace{\frac{1}{2}\|x - b \|^2_2}^{g(x)} + \overbrace{\lambda \|x\|_1}^{h(x)}}_{f(x)} \\

f'(x) &= (x - b) + \lambda \text{sgn}(x) = 0\\
x &= 
\left\{
\begin{aligned}
& b + \lambda,  &b < -\lambda \\
& 0,  &b \leq |\lambda| \\
& b - \lambda, &b > \lambda
\end{aligned}
\right.


\end{align*}
$$
**Noted as Soft Thresholding function**
$$
\begin{align*}
x &= \arg \min_x \underbrace{\overbrace{\frac{1}{2}\| Ax - b \|^2_2}^{g(x)} + \overbrace{\lambda \|x\|_1}^{h(x)}}_{f(x)} \\
x &= S_{\lambda}(b)
\end{align*}
$$



$$
\begin{align*}
x &= \arg \min_x \underbrace{\overbrace{\frac{1}{2}\| Ax - b \|^2_2}^{g(x)} + \overbrace{\lambda \|x\|_1}^{h(x)}}_{f(x)} \\
g(x) &\approx g(x_0) + \nabla g(x_0)(x-x_0) + \frac{\nabla^2g(x_0)}{2}(x - x_0)^2 \\

&= g(x_0) + \nabla g(x_0)(x-x_0) + \frac{1}{2t}(x - x_0)^2 \\

\end{align*}
$$
**Then,** $g(x_0)$ is constant, remove and add some terms
$$
\begin{align*}
x &= \arg \min_x \nabla g(x_0)(x-x_0) + \frac{1}{2t}(x - x_0)^2 + \lambda \|x\|_1 \\


&= \arg \min_x \frac{1}{2t} \left [(x-x_0)^2 + 2t\nabla g(x_0)(x-x_0) + (t\nabla g(x_0))^2 \right ] + \lambda \|x\|_1 \\

&= \arg \min_x \frac{1}{2t} \left [x-x_0 + t\nabla g(x_0) \right ]^2 + \lambda \|x\|_1 \\

&= \arg \min_x \frac{1}{2t} \left \|x - (x_0 - t\nabla g(x_0)) \right \|^2_2 + \lambda \|x\|_1 \\

\end{align*}
$$
**set** $z = x_0 - t \nabla g(x_0)$
$$
\begin{align*}
x &= \arg \min_x \underbrace{\overbrace{\frac{1}{2}\| Ax - b \|^2_2}^{g(x)} + \overbrace{\lambda \|x\|_1}^{h(x)}}_{f(x)} \\

x &= \arg \min_x \frac{1}{2t}\|x - z \|^2_2 + \lambda \|x\|_1 \\

x^* &= \arg \min_x \frac{1}{2}\|x - z \|^2_2 + \lambda  t \|x\|_1  = S_{\lambda t}(z) \\

\mathrm{prox}_{t, h(\cdot)}(z) &= \arg \min_x \frac{1}{2}\|x - z \|^2_2 + t \cdot h(x) 


\end{align*}
$$
**Iteration :**
$$
\begin{align*}

x &= \arg \min_x \underbrace{\overbrace{\frac{1}{2}\| Ax - b \|^2_2}^{g(x)} + \overbrace{\lambda \|x\|_1}^{h(x)}}_{f(x)} \\

z^{(k)} &= x^{(k)} - t \nabla g(x^{(k)}) \\

x^{(k+1)} &= \mathrm{prox}_{t, h(\cdot)}(z^{(k)}) \\

&= \arg \min_x \frac{1}{2t}\|x - z^{(k)} \|^2_2 + \lambda  \|x\|_1
\end{align*}
$$

---

---


$$
F_2(w) = \frac{1}{2}\|Aw - b \|^2_2 + \lambda \|w\|_1
$$





$$
F_2(w) &= \overbrace{\frac{1}{2}\| Aw - b \|^2_2}^{f_2(w)} + \overbrace{\lambda \|w\|_1}^{g_2(w)} \\

\nabla F_2(w) &= A^{\mathsf{T}}(Aw - b) + \lambda \text{sgn}(w) \\

\mathrm{prox}_{g_2}(x) &= \arg \min_w \overbrace{\frac{1}{2}\|w - x \|^2_2}^{f_2(w)} + \overbrace{\lambda \|w\|_1}^{g_2(w)} \\
$$






























