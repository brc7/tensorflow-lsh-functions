This repository implements a variety of locality sensitive hash functions in
Tensorflow 2.0.

# Usage

The hash functions are implemented in a single module, ``lsh_functions.py``.
We currenly support LSH in several important metric spaces, including
the Euclidean, Manhattan and Lp norm spaces (``PStableHash``), integer and
string token spaces (``IntMinHash`` and ``StringMinHash``), and inner product
spaces (``SRPHash``).

You can access these classes by simply importing the LSH function module.

### Dependencies

The only dependencies are Python3 and Tensorflow 2.0. The tests use numpy, but
this is not required to use the module.

### Tests

To run the tests, make sure you have Tensorflow and Numpy installed and run:
``python3 lsh_functions_test.py``

### Contributing
If you find a bug, feel free to create an issue. Ideally, this will include
your Python version, Tensorflow version and a self-contained, minimal program
to reproduce the bug.

If you would like to request a new LSH function or other functionality,
create an issue (or create a pull request with the desired functionality).

# License

This repository is free for research and commercial use, under the Apache-2
license. However, if you find this repo useful, please give us a star or -
even better - [get in touch](https://randorithms.com/about)!

