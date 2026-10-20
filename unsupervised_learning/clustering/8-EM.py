#!/usr/bin/env python3
"""Expectation Maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the EM algorithm for a GMM"""
    fail = (None, None, None, None, None)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return fail
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return fail
    if not isinstance(iterations, int) or iterations <= 0:
        return fail
    if not isinstance(tol, (int, float)) or tol < 0:
        return fail
    if not isinstance(verbose, bool):
        return fail

    pi, m, S = initialize(X, k)
    if pi is None:
        return fail
    g, lik = expectation(X, pi, m, S)
    if g is None:
        return fail
    if verbose:
        print("Log Likelihood after 0 iterations: {}".format(round(lik, 5)))
    prev = lik
    i = 0
    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        g, lik = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(lik, 5)))
        if abs(lik - prev) <= tol:
            break
        prev = lik
    if verbose and i % 10 != 0:
        print("Log Likelihood after {} iterations: {}".format(
            i, round(lik, 5)))
    return pi, m, S, g, lik
