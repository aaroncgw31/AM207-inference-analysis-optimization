import numpy as np
import pandas as pd
import scipy


from numpy.random import gamma, randint, uniform
from scipy.stats import multivariate_normal


num_sweeps = 100000
a = 9
b = 6
c = 9
d = 0.00001
alpha = 5.3  # 1.54517349188
beta = 475000  # 133514.171939
normal_cov = np.array([[0.1, 0], [0, 1000]])
burn_in = .1
thinning = 100

kcancer = pd.read_csv('kcancer.csv')
num_counties = len(kcancer)
n = kcancer['pop'].values
y = kcancer['dc'].values
theta = (y+1.)/n
accepted_samples = []


def calculate_log_p_ratio(alpha_beta_proposed, alpha_beta_old, theta):
    alpha_p, beta_p = alpha_beta_proposed
    alpha_o, beta_o = alpha_beta_old

    # print('Proposed: ', alpha_beta_proposed)
    # print('Old: ', alpha_beta_old)

    term_1 = num_counties*(alpha_p*np.log(beta_p) +
                           np.log(scipy.special.gamma(alpha_o)) -
                           alpha_o*np.log(beta_o) -
                           np.log(scipy.special.gamma(alpha_p)))
    # print('Term 1: ', term_1)
    term_2 = np.sum((alpha_p - alpha_o) * np.log(theta))
    # print('Term 2: ', term_2)
    term_3 = (a - 1) * np.log(alpha_p / alpha_o)
    # print('Term 3: ', term_3)
    term_4 = (c - 1) * np.log(beta_p / beta_o)
    # print('Term 4: ', term_4)
    term_5 = -b * (alpha_p - alpha_o) - d * (beta_p - beta_o) - (beta_p - beta_o) * np.sum(theta)
    # print('Term 5: ', term_5)
    log_p_ratio = term_1 + term_2 + term_3 + term_4 + term_5
    return log_p_ratio


for sweep in range(num_sweeps):

    print('Sweep Number: ', sweep)

    # choose random j and sample theta_j
    j = randint(low=0, high=num_counties)
    theta[j] = gamma(shape=alpha + y[j], scale=1. / (beta + 5 * n[j]))

    # sample candidate alpha, beta
    alpha_beta_old = np.array([alpha, beta])
    alpha_beta_proposed = multivariate_normal.rvs(
        mean=alpha_beta_old,
        cov=normal_cov)

    # calculate alpha, beta probabilities under proposal distribution
    #     q_old_given_proposed = multivariate_normal.pdf(
    #         x=alpha_beta_old,
    #         mean=alpha_beta_proposed,
    #         cov=normal_cov)
    #     q_proposed_given_old = multivariate_normal.pdf(
    #         x=alpha_beta_proposed,
    #         mean=alpha_beta_old,
    #         cov=normal_cov)

    # calculate unnormalized alpha, beta probabilities
    log_p_ratio = calculate_log_p_ratio(
        alpha_beta_proposed=alpha_beta_proposed,
        alpha_beta_old=alpha_beta_old,
        theta=theta)

    # calculate criterion (called alpha in the lecture notes)
    # q's actually cross out
    #     log_criterion = np.log(p_unnormalized_proposed) - np.log(q_proposed_given_old) \
    #         - np.log(p_unnormalized_old) + np.log(q_old_given_proposed)
    p_ratio = np.exp(log_p_ratio)
    if np.isinf(p_ratio):
        criterion = 1
    else:
        criterion = min(1., p_ratio)
    print('Criterion: ', criterion)

    # sample uniform for acceptance/rejection
    if uniform() < criterion:
        print('Accepted Proposal')
        alpha, beta = alpha_beta_proposed
    else:
        print('Rejected Proposal')
    accepted_samples.append((alpha, beta))

accepted_samples = np.array(accepted_samples)
sweep_number = np.arange(num_sweeps)

# burn in i.e. drop first 10%
# starting_index = int(num_sweeps * burn_in)
# accepted_samples = accepted_samples[starting_index:]
# sweep_number = sweep_number[starting_index:]

# thin every 10 steps
accepted_samples = accepted_samples[::thinning]
sweep_number = sweep_number[::thinning]


import plotly
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

data = [
    go.Scatter(x=sweep_number,
               y=accepted_samples[:, 0],
               mode="lines",
               name="alpha"),
    ]

layout = dict(
    title="Alpha by Sweep Number",
    xaxis=dict(title="Sweep Number"),
    yaxis=dict(title="Alpha")
)

fig = go.Figure(
    data=data,
    layout=layout)
fig.show()


data = [
    go.Scatter(x=sweep_number,
               y=accepted_samples[:, 1],
               mode="lines",
               name="beta"),
    ]

layout = dict(
    title="Beta by Sweep Number",
    xaxis=dict(title="Sweep Number"),
    yaxis=dict(title="Beta")
)

fig = go.Figure(
    data=data,
    layout=layout)
fig.show()

print(10)