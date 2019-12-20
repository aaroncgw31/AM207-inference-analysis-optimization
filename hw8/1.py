import numpy
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.optimizers import adam, sgd


def black_box_variational_inference(logprob, D, num_samples):
    """
    Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557
    code taken from:
    https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py
    """

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:]
        return mean, log_std

    def gaussian_entropy(log_std):
        return 0.5 * D * (1.0 + np.log(2 * np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)

    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std = unpack_params(params)
        samples = rs.randn(num_samples, D) * np.exp(log_std) + mean
        lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples, t))
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


def variational_inference(Sigma_W, y_train, x_train, S, max_iteration, step_size, verbose):
    '''implements wrapper for variational inference via bbb for bayesian regression'''
    D = Sigma_W.shape[0]
    Sigma_W_inv = np.linalg.inv(Sigma_W)
    Sigma_W_det = np.linalg.det(Sigma_W)
    variational_dim = D

    # define the log prior on the model parameters
    def log_prior(W):
        constant_W = -0.5 * (D * np.log(2 * np.pi) + np.log(Sigma_W_det))
        exponential_W = -0.5 * np.diag(np.dot(np.dot(W, Sigma_W_inv), W.T))
        log_p_W = constant_W + exponential_W
        return log_p_W

    # define the log likelihood
    def log_lklhd(W):
        log_odds = np.matmul(W, x_train) + 10
        p = 1 / (1 + np.exp(-log_odds))
        log_likelihood = y_train * np.log(p)
        return log_likelihood

    # define the log joint density
    log_density = lambda w, t: log_lklhd(w) + log_prior(w)

    # build variational objective.
    objective, gradient, unpack_params = black_box_variational_inference(log_density, D, num_samples=S)

    def callback(params, t, g):
        if verbose:
            if verbose:
                if t % 10 == 0:
                    var_means = params[:D]
                    var_variance = np.diag(np.exp(params[D:]) ** 2)
                    print("Iteration {} lower bound {}; gradient mag: {}".format(
                        t, -objective(params, t), np.linalg.norm(gradient(params, t))))
                    print('Variational Mean: ', var_means)
                    print('Variational Variances: ', var_variance)
    print("Optimizing variational parameters...")
    # initialize variational parameters
    init_mean = 0 * np.ones(D)
    init_log_std = -1 * np.ones(D)
    init_var_params = np.concatenate([init_mean, init_log_std])

    # perform gradient descent using adam (a type of gradient-based optimizer)
    variational_params = adam(gradient, init_var_params, step_size=step_size, num_iters=max_iteration,
                              callback=callback)

    return variational_params


def variational_bernoulli_regression(Sigma_W, x_train, y_train, S=2000, max_iteration=2000, step_size=1e-2,
                                     verbose=True):

    D = Sigma_W.shape[0]

    # approximate posterior with mean-field gaussian
    variational_params = variational_inference(
        Sigma_W=Sigma_W,
        y_train=y_train,
        x_train=x_train,
        S=S,
        max_iteration=max_iteration,
        step_size=step_size,
        verbose=verbose)

    # sample from the variational posterior
    var_means = variational_params[:D]
    var_variance = np.diag(np.exp(variational_params[D:]) ** 2)

    return var_variance, var_means


ys = np.array([[1.]])
xs = np.array([[-20]])
N = 1
D = 1
Sigma_W = np.eye(D)


print(variational_bernoulli_regression(
    Sigma_W=Sigma_W,
    x_train=xs,
    y_train=ys,
    S=4000,
    max_iteration=2000,
    step_size=1e-1,
    verbose=True))