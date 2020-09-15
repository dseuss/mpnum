![mpnum](docs/mpnum_logo_144.png)
=====


## A matrix product representation library for Python

[![JOSS](http://joss.theoj.org/papers/f5d6dc694fffcffa13f0def4b42bb113/status.svg)](http://joss.theoj.org/papers/f5d6dc694fffcffa13f0def4b42bb113)
[![PyPI](https://img.shields.io/pypi/v/mpnum.svg)](https://pypi.python.org/pypi/mpnum/)
[![Travis](https://travis-ci.org/dseuss/mpnum.svg?branch=master)](https://travis-ci.org/dseuss/mpnum)
[![Documentation Status](https://readthedocs.org/projects/mpnum/badge/?version=latest)](http://mpnum.readthedocs.org/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/dseuss/mpnum/badge.svg?branch=master)](https://coveralls.io/github/dseuss/mpnum?branch=master)
[![Maintainability](https://api.codeclimate.com/v1/badges/4b06c328d4df622ade65/maintainability)](https://codeclimate.com/github/dseuss/mpnum/maintainability)

mpnum is a flexible, user-friendly, and expandable toolbox for the matrix product state/tensor train tensor format. mpnum provides:

* support for well-known matrix product representations, such as:
  * matrix product states ([MPS](http://mpnum.readthedocs.org/en/latest/intro.html#matrix-product-states-mps)), also known as tensor trains (TT)
  * matrix product operators ([MPO](http://mpnum.readthedocs.org/en/latest/intro.html#matrix-product-operators-mpo))
  * local purification matrix product states ([PMPS](http://mpnum.readthedocs.org/en/latest/intro.html#local-purification-form-mps-pmps))
  * arbitrary matrix product arrays ([MPA](http://mpnum.readthedocs.org/en/latest/intro.html#matrix-product-arrays))
* arithmetic operations: addition, multiplication, contraction etc.
* [compression](http://mpnum.readthedocs.org/en/latest/mpnum.html#mpnum.mparray.MPArray.compress), [canonical forms](http://mpnum.readthedocs.org/en/latest/mpnum.html#mpnum.mparray.MPArray.canonicalize), etc.
* finding [extremal eigenvalues](http://mpnum.readthedocs.org/en/latest/mpnum.html#mpnum.linalg.eig) and eigenvectors of MPOs (DMRG)
* flexible tools for new matrix product algorithms

To install the latest stable version run

    pip install mpnum

If you want to install this fork of `mpnum` from source, please run (on Unix)

    git clone https://github.com/moritzlange/mpnum.git
    cd mpnum
    pip install .

In order to run the tests and build the documentation, you have to install the development dependencies via

    pip install -r requirements.txt

For more information, see:

* [Introduction to mpnum](http://mpnum.readthedocs.org/en/latest/intro.html)
* [Notebook with code examples](examples/mpnum_intro.ipynb)
* [More examples from quantum physics](https://github.com/milan-hl/mpnum-examples/) (ground states, time evolution, unitary circuits)
* [Library reference](http://mpnum.readthedocs.org/en/latest/)
* [Contribution Guidelines](http://mpnum.readthedocs.io/en/latest/devel.html)

Required packages:

* six, numpy, scipy

Supported Python versions:

* 2.7, 3.4, 3.5, 3.6

Alternatives:

* [TT-Toolbox](https://github.com/oseledets/TT-Toolbox) for Matlab
* [ttpy](https://github.com/oseledets/ttpy) for Python
* [ITensor](https://github.com/ITensor/ITensor) for C++
* [t3f](https://github.com/Bihaqo/t3f) for TensorFlow
* [Alps](https://github.com/ALPSCore/ALPSCore) for C++
* [uni10](https://gitlab.com/uni10/uni10/) for C++


## How to contribute
Contributions of any kind are very welcome.
Please use the [issue tracker](https://github.com/dseuss/mpnum/issues) for bug reports.
If you want to contribute code, please see the [section on how to contribute](http://mpnum.readthedocs.io/en/latest/devel.html) in the documentation.


## Contributors

* Daniel Suess, <daniel@dsuess.me>, [University of Cologne](http://www.thp.uni-koeln.de/gross/)
* Milan Holzaepfel, <mail@mholzaepfel.de>, [Ulm University](http://qubit-ulm.com/)


## License

Distributed under the terms of the BSD 3-Clause License (see [LICENSE](LICENSE)).


## Citations

If you use `mpnum` for yor paper, please cite:

> Suess, Daniel and Milan Holzäpfel (2017). mpnum: A matrix product representation library for Python. Journal of Open Source Software, 2(20), 465, https://doi.org/10.21105/joss.00465

* BibTeX: [mpnum.bib](paper/mpnum.bib)

mpnum has been used and cited in the following publications:

* I. Dhand et al. (2017), [arXiv 1710.06103](https://arxiv.org/abs/1710.06103)
* I. Schwartz, J. Scheuer et al. (2017), [arXiv 1710.01508](https://arxiv.org/abs/1710.01508)
* J. Scheuer et al. (2017), [arXiv 1706.01315](https://arxiv.org/abs/1706.01315)
* B. P. Lanyon, Ch. Maier et al, [Nature Physics 13, 1158–1162 (2017)](https://doi.org/10.1038/nphys4244), [arXiv 1612.08000](https://arxiv.org/abs/1612.08000)
