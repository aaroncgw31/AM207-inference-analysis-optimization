import numpy
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.optimizers import adam


class Feedforward:
    def __init__(self, architecture, random=None, weights=None):
        self.params = {'H': architecture['width'],
                       'L': architecture['hidden_layers'],
                       'D_in': architecture['input_dim'],
                       'D_out': architecture['output_dim'],
                       'activation_type': architecture['activation_fn_type'],
                       'activation_params': architecture['activation_fn_params']}

        self.D = ((architecture['input_dim'] * architecture['width'] + architecture['width'])
                  + (architecture['output_dim'] * architecture['width'] + architecture['output_dim'])
                  + (architecture['hidden_layers'] - 1) * (architecture['width'] ** 2 + architecture['width'])
                  )

        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        self.h = architecture['activation_fn']

        if weights is None:
            self.weights = self.random.normal(0, 1, size=(1, self.D))
        else:
            self.weights = weights

        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))

    def forward(self, weights, x):

        ''' Forward pass given weights and input '''
        H = self.params['H']
        D_in =  self.params['D_in']
        D_out = self.params['D_out']

        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            assert x.shape[0] == D_in
            x = x.reshape((1, D_in, -1))
        else:
            assert x.shape[1] == D_in

        weights = weights.T

        # input to first hidden layer
        W = weights[:H * D_in].T.reshape((-1, H, D_in))
        b = weights[H * D_in:H * D_in + H].T.reshape((-1, H, 1))
        input = self.h(np.matmul(W, x) + b)
        index = H * D_in + H

        assert input.shape[1] == H

        # additional hidden layers
        for _ in range(self.params['L'] - 1):
            before = index
            W = weights[index:index + H * H].T.reshape((-1, H, H))
            index += H * H
            b = weights[index:index + H].T.reshape((-1, H, 1))
            index += H
            output = np.matmul(W, input) + b
            input = self.h(output)

            assert input.shape[1] == H

        # output layer
        W = weights[index:index + H * D_out].T.reshape((-1, D_out, H))
        b = weights[index + H * D_out:].T.reshape((-1, D_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params['D_out']
        # output = output.squeeze(1)  # Rylan added
        return output

    def make_objective(self, x_train, y_train, reg_param=None):
        ''' Make objective functions: depending on whether or not you want to apply l2 regularization '''

        if reg_param is None:

            def objective(W, t):
                squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1) ** 2
                sum_error = np.sum(squared_error)
                return sum_error

            return objective, grad(objective)

        else:

            def objective(W, t):
                squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1) ** 2
                mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W)
                return mean_error

            return objective, grad(objective)

    def fit(self, x_train, y_train, params, reg_param=None):
        ''' Wrapper for MLE through gradient descent '''
        assert x_train.shape[0] == self.params['D_in']
        assert y_train.shape[0] == self.params['D_out']

        ### make objective function for training
        self.objective, self.gradient = self.make_objective(x_train, y_train, reg_param)

        ### set up optimization
        step_size = 0.01
        max_iteration = 5000
        check_point = 100
        weights_init = self.weights.reshape((1, -1))
        mass = None
        optimizer = 'adam'
        random_restarts = 5

        if 'step_size' in params.keys():
            step_size = params['step_size']
        if 'max_iteration' in params.keys():
            max_iteration = params['max_iteration']
        if 'check_point' in params.keys():
            self.check_point = params['check_point']
        if 'init' in params.keys():
            weights_init = params['init']
        if 'call_back' in params.keys():
            call_back = params['call_back']
        if 'mass' in params.keys():
            mass = params['mass']
        if 'optimizer' in params.keys():
            optimizer = params['optimizer']
        if 'random_restarts' in params.keys():
            random_restarts = params['random_restarts']

        def call_back(weights, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if iteration % check_point == 0:
                print("Iteration {} lower bound {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(
                    self.gradient(weights, iteration))))

        ### train with random restarts
        optimal_obj = 1e16
        optimal_weights = self.weights

        for i in range(random_restarts):
            if optimizer == 'adam':
                adam(self.gradient, weights_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[-100:])

            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights = self.weight_trace[-100:][opt_index].reshape((1, -1))
            weights_init = self.random.normal(0, 1, size=(1, self.D))

        self.objective_trace = self.objective_trace[1:]
        self.weight_trace = self.weight_trace[1:]


###define rbf activation function
alpha = 1
c = 0
h = lambda x: np.exp(-alpha * (x - c)**2)

###neural network model design choices
width = 5
hidden_layers = 1
input_dim = 1
output_dim = 1

architecture = {'width': width,
               'hidden_layers': hidden_layers,
               'input_dim': input_dim,
               'output_dim': output_dim,
               'activation_fn_type': 'rbf',
               'activation_fn_params': 'c=0, alpha=1',
               'activation_fn': h}

params = {'step_size':1e-3,
          'max_iteration': 6001,
          'random_restarts':1,
          'check_point':200}

#set random state to make the experiments replicable
rand_state = 0
random = np.random.RandomState(rand_state)

#instantiate a Feedforward neural network object
nn = Feedforward(architecture, random=random)

import pandas as pd


df = pd.read_csv('HW7_data.csv')
print(df.head())
xs = df['x'].values
ys = df['y'].values
nn.fit(xs.reshape((1, -1)), ys.reshape((1, -1)), params)


# adapted from Lecture 15
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


# adapted from lecture 15
def variational_nn_inference(Sigma_W, y_train, x_train, S, max_iteration, step_size, verbose):
    '''implements wrapper for variational inference via bbb for bayesian regression'''
    D = Sigma_W.shape[0]
    Sigma_W_inv = np.linalg.inv(Sigma_W)
    Sigma_W_det = np.linalg.det(Sigma_W)
    variational_dim = D

    # define the log prior on the model parameters
    #     def log_prior(W):
    #         constant_W = -0.5 * (D * np.log(2 * np.pi) + np.log(Sigma_W_det))
    #         exponential_W = -0.5 * np.diag(np.dot(np.dot(W, Sigma_W_inv), W.T))
    #         log_p_W = constant_W + exponential_W
    #         return log_p_W

    # define the log likelihood
    #     def log_lklhd(W):
    #         log_odds = np.matmul(W, x_train) + 10
    #         p = 1 / (1 + np.exp(-log_odds))
    #         log_likelihood = y_train * np.log(p)
    #         return log_likelihood

    def log_joint(W, ys, xs):
        term1 = -0.5 * W @ W.T / 25
        yhats = nn.forward(W, xs.reshape(1, -1))
        term2 = -0.5 * 4 * np.square(yhats-ys).sum(0).mean()
        return term1 + term2

    # define the log joint density
    #     log_density = lambda w, t: log_lklhd(w) + log_prior(w)
    log_density = lambda w, t: log_joint(w, xs=x_train, ys=y_train)

    # build variational objective.
    objective, gradient, unpack_params = black_box_variational_inference(log_density, D, num_samples=S)

    def callback(params, t, g):
        if verbose:
            if verbose:
                if t % 100 == 0:
                    var_means = params[:D]
                    var_variance = np.exp(params[D:]) ** 2
                    # var_variance = np.diag(np.exp(params[D:]) ** 2)
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

D = 16
Sigma_W = np.eye(D)
num_samples = 4000
max_iterations = 2501
step_size = 1e-2

variational_params = variational_nn_inference(
        Sigma_W=Sigma_W,
        y_train=ys,
        x_train=xs,
        S=num_samples,
        max_iteration=max_iterations,
        step_size=step_size,
        verbose=True)
var_means = variational_params[:D]
var_variance = np.diag(np.exp(variational_params[D:])**2)


# sample from the variational posterior
posterior_sample_size = 100
posterior_samples = numpy.random.multivariate_normal(var_means, var_variance, size=posterior_sample_size)

# sample from posterior predictive
x_test = numpy.linspace(-8, 8, 100)
y_tests, x_tests = [], []
for posterior_sample in posterior_samples:
    y_test = nn.forward(
        posterior_sample.reshape((1, -1)),
        x_test.reshape((1, -1))).squeeze()
    y_test += numpy.random.normal(loc=0, scale=0.5, size=y_test.shape)
    x_tests.append(x_test)
    y_tests.append(y_test)
x_tests = numpy.concatenate(x_tests)
y_tests = numpy.concatenate(y_tests)

y_mle = nn.forward(nn.weights, x_test.reshape((1, -1))).squeeze()


from IPython.display import Image
import plotly.graph_objects as go
import plotly.io as pio

plot_data = [
    go.Scatter(x=x_tests, y=y_tests, mode='markers', name='Posterior Predictive'),
    go.Scatter(x=x_test, y=y_mle, name='MLE'),
    go.Scatter(x=xs, y=ys, mode='markers', name='Original Data')]
fig = go.Figure(data=plot_data)
# fig.show()
Image(pio.to_image(fig, format='png'))
