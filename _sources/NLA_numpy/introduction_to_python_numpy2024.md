

## Matrix and Vector Manipulation
Numpy also implements the basic operations on matrix and vectors. In particular, the following functions can be useful in this course:

* `np.linalg.norm(a, p)`: Computes the $$p$$-norm of a vector or a matrix $$a$$;
* `np.linalg.cond(A, p)`: Computes the condition number in $p$-norm of a matrix $$A$$;
* `np.linalg.matrix_rank(A)`: Computes the rank of the matrix $$A$$;
* `np.linalg.inv(A)`: When invertible, compute the inverse matrix of $$A$$. _Warning:_ Very slow;
* `np.transpose(A)`: Compute the transpose matrix of $$A$$. It is equivalent to `A.T`;
* `np.reshape(a, new_shape)`: Reshape an array `a` to a given shape.

## Read data with pandas
Since we will frequently work with data, it will be important to be able to manipulate them. In this class, we will learn how to load a dataset into Python by using a library called `pandas`, whose documentation can be found [here](https://pandas.pydata.org/docs/user_guide/index.html#user-guide). 

As an example, download the data from Kaggle at the following link: [www.kaggle.com/mysarahmadbhat/us-births-2000-to-2014](https://www.kaggle.com/mysarahmadbhat/us-births-2000-to-2014). 

Then, place it in the same folder as the Python file on which you are working and use the following code to load it in memory.

```
import pandas as pd

# Read data from a csv file
data = pd.read_csv('./data/US_births_2000-2014_SSA.csv')
```

Pandas uses similar function name as numpy to keep everything coherent. For example, we can check the shape of `data` by using the function `print(data.shape)`. Moreover, a pandas dataframe can be casted into a numpy array by simply

```
import numpy as np

# Cast into numpy array
np_data = np.array(data)

# Check that the dimension didn't change
print(f"{data.shape} should be equal to {np_data.shape}")
```