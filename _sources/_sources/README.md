# Statistical and Matematical Methods for Machine Learning

This is the official repository for the notes related to the course of Statistical and Mathematical Methods for Machine Learning (SMM), at the Master Degree in Artifical Intelligence, University of Bologna (BO). The updated version refers to the Academic Year 2024-2025.

To check the compiled version of this code, please go to https://devangelista2.github.io/statistical-mathematical-methods.

## Edit the notes
To edit the notes, after cloning the repository with:

```
git clone https://github.com/devangelista2/statistical-mathematical-methods.git
```

you can just access the `.ipynb` files referring to the topic of interest, and edit it by any code editor which works with Python Notebooks.

## Compile
To compile the updated version of the notes, please install `jupyter-book` and `ghp` packages on your virtual environment, via:

```
pip install -U jupyter-book
pip install ghp-import
```

Then move to the directory corresponding to the repository:

```
cd statistical-mathematical-methods
```

and compile the code with:

```
jupyter-book build .
```

The compiled HTML files can be found inside the `build` folder. To push the updated version online, run:

```
ghp-import -n -p -f _build/html
```

Please note that this operation **DOESN'T** push the code that compiles to the website, but only the website itself. Remember to also commit and push the source code as:

```
git add .
git commit -m "<CHANGELOG>"
git push
```

Check https://jupyterbook.org/ for a complete Documentation on `jupyter-book`.

## Suggestions and Reporting
Please report any encountered error or any suggestion to my email address: davide.evangelista5@unibo.it.