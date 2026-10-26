"""Gaussian-emission Hidden Markov Model implemented from scratch with numpy.

Provides:
  - GaussianHMM.init_supervised : estimate pi, A, B (means/variances) from labelled sequences
  - GaussianHMM.fit             : Baum-Welch (EM) over multiple sequences with a
                                  log-likelihood convergence check (|dLL| < tol)
  - GaussianHMM.viterbi         : most likely hidden-state path (log space)
  - GaussianHMM.score           : log-likelihood log P(X | model) via the scaled forward pass
"""
import numpy as np


class GaussianHMM:
    def __init__(self, n_states, n_features, var_floor=1e-3, trans_prior=1.0):
        self.K = n_states
        self.D = n_features
        self.var_floor = var_floor      # keeps emission variances from collapsing to zero
        self.trans_prior = trans_prior  # Dirichlet pseudo-count added to transition counts
        self.pi = np.full(n_states, 1.0 / n_states)
        self.A = np.full((n_states, n_states), 1.0 / n_states)
        self.means = np.zeros((n_states, n_features))
        self.vars = np.ones((n_states, n_features))
        self.history = []

    # ---------- emission model B: diagonal Gaussian per state ----------
    def log_emission(self, X):
        """Return log p(x_t | z_t = k) as a (T, K) matrix."""
        diff = X[:, None, :] - self.means[None, :, :]
        return -0.5 * (np.sum(np.log(2 * np.pi * self.vars), axis=1)[None, :]
                       + np.sum(diff ** 2 / self.vars[None, :, :], axis=2))

    @staticmethod
    def _emission_probs(logB):
        # subtract the per-timestep max for stability; the shift is returned so the
        # log-likelihood can be corrected
        shift = logB.max(axis=1, keepdims=True)
        return np.exp(logB - shift), shift.ravel()

    # ---------- supervised initialisation ----------
    def init_supervised(self, seqs, labels):
        """Count-based estimates of pi, A and Gaussian means/variances from labelled data."""
        X, Z = np.vstack(seqs), np.concatenate(labels)
        for k in range(self.K):
            self.means[k] = X[Z == k].mean(axis=0)
            self.vars[k] = np.maximum(X[Z == k].var(axis=0), self.var_floor)
        pi = np.full(self.K, self.trans_prior)
        A = np.full((self.K, self.K), self.trans_prior)
        for z in labels:
            pi[z[0]] += 1
            np.add.at(A, (z[:-1], z[1:]), 1)
        self.pi = pi / pi.sum()
        self.A = A / A.sum(axis=1, keepdims=True)
        return self

    # ---------- forward-backward (scaled) ----------
    def _forward_backward(self, X):
        T = len(X)
        B, shift = self._emission_probs(self.log_emission(X))
        alpha = np.zeros((T, self.K))
        c = np.zeros(T)
        alpha[0] = self.pi * B[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        beta = np.ones((T, self.K))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        loglik = np.sum(np.log(c)) + shift.sum()
        return alpha, beta, c, B, loglik

    def score(self, X):
        return self._forward_backward(X)[4]

    def posteriors(self, X):
        alpha, beta, *_ = self._forward_backward(X)
        g = alpha * beta
        return g / g.sum(axis=1, keepdims=True)

    # ---------- Baum-Welch ----------
    def fit(self, seqs, tol=1e-4, max_iter=200, verbose=False):
        """EM over a list of observation sequences.

        Stops when the per-window average log-likelihood improves by less than `tol`
        (max_iter is only a safety cap). Returns self; self.history holds the LL curve.
        """
        n_obs = sum(len(X) for X in seqs)
        self.history = []
        for it in range(max_iter):
            pi_acc = np.zeros(self.K)
            A_acc = np.full((self.K, self.K), self.trans_prior)
            g_sum = np.zeros(self.K)
            x_sum = np.zeros((self.K, self.D))
            x2_sum = np.zeros((self.K, self.D))
            total_ll = 0.0
            # E-step
            for X in seqs:
                alpha, beta, c, B, ll = self._forward_backward(X)
                total_ll += ll
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)
                # xi_t(i,j) = alpha_t(i) A_ij B_{t+1}(j) beta_{t+1}(j) / c_{t+1}
                xi = (alpha[:-1, :, None] * self.A[None]
                      * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
                pi_acc += gamma[0]
                A_acc += xi.sum(axis=0)
                g_sum += gamma.sum(axis=0)
                x_sum += gamma.T @ X
                x2_sum += gamma.T @ (X ** 2)
            # M-step
            # same Dirichlet smoothing as A: with few sequences, pi would otherwise
            # collapse onto whichever states happened to start them
            self.pi = (pi_acc + self.trans_prior) / (pi_acc + self.trans_prior).sum()
            self.A = A_acc / A_acc.sum(axis=1, keepdims=True)
            self.means = x_sum / g_sum[:, None]
            self.vars = np.maximum(x2_sum / g_sum[:, None] - self.means ** 2, self.var_floor)

            avg_ll = total_ll / n_obs
            self.history.append(avg_ll)
            if verbose:
                print(f"iter {it:3d}  avg log-likelihood {avg_ll:.6f}")
            if it > 0 and abs(self.history[-1] - self.history[-2]) < tol:
                break
        self.converged_ = it < max_iter - 1
        return self

    # ---------- Viterbi ----------
    def viterbi(self, X):
        """Return the most likely state path (int array of length T) and its log-probability."""
        T = len(X)
        logB = self.log_emission(X)
        logA = np.log(self.A)
        delta = np.zeros((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)
        delta[0] = np.log(self.pi) + logB[0]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + logA      # cand[i, j]: come from i into j
            psi[t] = cand.argmax(axis=0)
            delta[t] = cand.max(axis=0) + logB[t]
        path = np.zeros(T, dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, delta[-1].max()
