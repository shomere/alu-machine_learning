#!/usr/bin/env python3
"""Expectation Maximization for a Gaussian Mixture Model."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """
    Performs the expectation maximization algorithm for a GMM.

    Returns:
        pi, m, S, g, log_likelihood
        or (None, None, None, None, None) on failure.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if (not isinstance(tol, (float, int))) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    if pi is None:
        return None, None, None, None, None

    g, log_likelihood = expectation(X, pi, m, S)

    if g is None:
        return None, None, None, None, None

    previous = None

    for i in range(iterations):

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}"
                  .format(i, log_likelihood))

        if previous is not None:
            if abs(log_likelihood - previous) <= tol:

                if verbose and i % 10 != 0:
                    print("Log Likelihood after {} iterations: {:.5f}"
                          .format(i, log_likelihood))

                return pi, m, S, g, log_likelihood

        previous = log_likelihood

        pi, m, S = maximization(X, g)

        if pi is None:
            return None, None, None, None, None

        g, log_likelihood = expectation(X, pi, m, S)

        if g is None:
            return None, None, None, None, None

    if verbose:
        print("Log Likelihood after {} iterations: {:.5f}"
              .format(iterations, log_likelihood))

    return pi, m, S, g, log_likelihood
