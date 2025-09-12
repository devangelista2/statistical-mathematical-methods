## Optimization in Machine Learning
While working with Machine Learning (ML) it is common to have a dataset $$\mathbb{D} = \{ (x^{(i)}, y^{(i)}) \}_{i=1}^N$$ and a parametric function $$f_w(x)$$ whose specific shape depends on the task. As already cited, _training_ a Machine Learning model is basically an optimization problem, where we need to find parameters $$w$$ (known as *weights*), such that $$f_w(x_i) \approx y_i$$ for any $$i = 1, \dots, N$$. To do that, we usually consider a **loss function** (see <a href="{{ site.baseurl }}{% link _teaching/short_introduction_to_ML2024.md %}">the related post</a> for more details), which in this case depends on the weights $$w$$ and the dataset $$\mathbb{D}$$. We will indicate is as $$\ell(w; \mathbb{D})$$.

In most of the cases, $$\ell(w; \mathbb{D})$$ can be written as a sum of simpler components, each depending on a specific datapoint, i.e.

$$
	\ell(w; \mathbb{D}) = \sum_{i=1}^N \ell_i(w; x^{(i)}, y^{(i)})
$$

and the training optimization problem becomes

$$
	w^* = \arg\min_w \ell(w; \mathbb{D}) = \arg\min_w  \sum_{i=1}^N \ell_i(w; x^{(i)}, y^{(i)})
$$

Which can be solved by <a href="{{ site.baseurl }}{% link _teaching/gradient_descent.md %}">Gradient Descent</a>, as

$$
	\begin{cases}
		w_0 \in \mathbb{R}^d \\
		w_{k+1} = w_k - \alpha_k \nabla_w \ell(w_k; \mathbb{D}) = w_k - \alpha_k \sum_{i=1}^N \nabla_w \ell_i(w_k; x^{(i)}, y^{(i)})
	\end{cases}
$$

Where we used that $$\nabla_w \ell(w_k; \mathbb{D}) = \nabla_w \sum_{i=1}^N \ell_i(w_k; x^{(i)}, y^{(i)}) = \sum_{i=1}^N \nabla_w \ell_i(w_k; x^{(i)}, y^{(i)})$$.

Thus, to compute each iteration of Gradient Descent we need the gradient with respect to the weights of the objective functions, that can be computed by summing up the gradients of the independent functions $$\ell_i(w; x^{(i)}, y^{(i)})$$.

As an example, a common loss function in Machine Learning is the Mean Squared Error (MSE), defined by

$$
	\ell(w; \mathbb{D}) = \sum_{i=1}^N MSE(f_w(x_i), y_i)
$$

where

$$
	MSE(y, y') = \frac{|| y - y' ||_2^2}{n} \quad y, y' \in \mathbb{R}^n
$$

Here, computing $$\nabla_w \ell(w; \mathbb{D})$$ is not hard, since

$$
	\nabla_w \ell(w; \mathbb{D}) = \nabla_w \sum_{i=1}^N MSE(f_w(x_i), y_i) = \ell(w; \mathbb{D}) \sum_{i=1}^N \nabla_w MSE(f_w(x_i), y_i)
$$

but

$$
	\nabla_w MSE(f_w(x_i), y_i) = \frac{1}{n} \nabla_w || f_w(x_i) - y_i ||_2^2 = \frac{2}{n} \nabla_w f_w(x_i)^T (f_w(x_i) - y_i)
$$

by applying the chain rule. When $$\nabla_w f_w(x_i)$$ can be explicitly computed (it depends on the shape of $$f_w$$), then the gradient descent iteration to solve the training optimization problem can be implemented as

$$
	w_{k+1} = w_k - \alpha_k \underbrace{\frac{1}{N} \sum_{i=1}^N \frac{2}{n} \nabla_w f_w(x_i)^T (f_w(x_i) - y_i)}_{\nabla_w \ell(w; \mathbb{D})}
$$

## Stochastic Gradient Descent (SGD)
Unfortunately, even if it is easy to compute the gradient of $$\ell_i(w; x^{(i)}, y^{(i)})$$ for any $$i$$, when the number of samples $$N$$ is large (which is common in Machine Learning), the computation of the full gradient $$\nabla_w \ell(w; \mathbb{D})$$ is prohibitive, mostly because of memory limitations. For this reason, in such optimization problems, instead of using a standard GD algorithm, it is better using the Stochastic Gradient Descent (SGD) method. That is a variant of the classical GD where, instead of computing $$\nabla_w \ell(w; \mathbb{D}) = \sum_{i=1}^N \nabla_w \ell_i(w; x^{(i)}, y^{(i)})$$, the summation is reduced to a limited number of terms, called a \emph{batch}. The idea is the following:

* Given a number $$N_{batch} \ll N$$ (usually called `batch_size`), randomly extract a subdataset $$\mathcal{M}$$ with $$\|\mathcal{M}\| = N_{batch}$$ from $$\mathbb{D}$$. This set will be called a **batch**;
	
* Approximate the true gradient 

$$
\nabla_w \ell(w; \mathbb{D}) = \sum_{i=1}^N \nabla_w \ell_i(w; x^{(i)}, y^{(i)})
$$ 

with

$$
\nabla_w \ell(w; \mathcal{M}) = \sum_{i\in \mathcal{M}} \nabla_w \ell_i(w; x^{(i)}, y^{(i)})
$$
	
* Compute one single iteration of the GD algorithm 

$$ 
w_{k+1} = w_k - \alpha_k \nabla_w \ell(w; \mathcal{M});
$$

* Repeat until you have extracted the full dataset. Notice that the random sampling at each iteration is done without replacement.

Each iteration of the algorithm above is usually called *batch iteration*. When the whole dataset has been processed, we say that we completed an **epoch** of the SGD method. This algorithm should be repeated for a fixed number $$E$$ of epochs to reach convergence.


### Drawbacks of SGD
Unfortunately, one of the biggest drawbacks of SGD with respect to GD, is that now we cannot check the convergence anymore (since we can't obviously compute the gradient of $$\ell(w; \mathbb{D})$$ to check its distance from zero, as it is required for the first Stopping Criteria) and we can't use the backtracking algorithm, for the same reason. As a consequence, the algorithm will stop ONLY after reaching the fixed number of epochs, and we must set a good value for the step size $$\alpha_k$$ by hand. Those problems are partially solved by recent algorithms like SGD with Momentum, Adam, AdaGrad, ... whose study is beyond the scope of the course.