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