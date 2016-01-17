"""
This file is part of the InferenceOfVariance project.
Copyright 2016 David W. Hogg (SCDA).

# To-do
- Write ABC code.
"""

import numpy as np
from corner import corner

def make_fake_data(N):
    return Truth[0] + np.sqrt(Truth[1]) * np.random.normal(size=N)

def ln_pseudo_likelihood(stats, pars, variances):
    empiricalmean, empiricalvar = stats
    mean, var = pars
    meanvar, varvar = variances
    return (-0.5 * (empiricalmean - mean) ** 2 / meanvar +
            -0.5 * (empiricalvar  - var ) ** 2 / varvar)

def ln_correct_likelihood(data, pars):
    mean, var = pars
    return np.sum(-0.5 * (data - mean) ** 2 / var) - (0.5 * len(data) * np.log(var))

def ln_prior(pars):
    mean, var = pars
    if mean < prior_info[0]:
        return -np.Inf
    if mean > prior_info[1]:
        return -np.Inf
    if var  < prior_info[2]:
        return -np.Inf
    if var  > prior_info[3]:
        return -np.Inf
    return 0.

def ln_pseudo_posterior(pars, stats, variances):
    lnp = ln_prior(pars)
    if not np.isfinite(lnp):
        return -np.Inf
    return lnp + ln_pseudo_likelihood(stats, pars, variances)

def ln_correct_posterior(pars, data):
    lnp = ln_prior(pars)
    if not np.isfinite(lnp):
        return -np.Inf
    return lnp + ln_correct_likelihood(data, pars)

def mcmc_step(pars, lnpvalue, lnp, args):
    newpars = pars + stepsizes * np.random.normal(size=len(pars))
    newlnpvalue = lnp(newpars, *args)
    if (newlnpvalue - lnpvalue) > np.log(np.random.uniform()):
        return newpars, newlnpvalue
    return pars.copy(), lnpvalue

def mcmc(pars0, lnp, nsteps, args):
    pars = pars0.copy()
    lnpvalue = lnp(pars, *args)
    parss = np.zeros((nsteps, len(pars)))
    for k in range(nsteps):
        pars, lnpvalue = mcmc_step(pars, lnpvalue, lnp, args)
        parss[k,:] = pars
    return parss

def main(N):
    Nstr = "{:04d}".format(N)

    print("main: making fake data")
    np.random.seed(23)
    data = make_fake_data(N)
    empiricalmean = np.mean(data)
    empiricalvar = np.sum((data - empiricalmean) ** 2) / (N - 1)
    stats = np.array([empiricalmean, empiricalvar])
    print(data, stats)

    print("main: running correct MCMC")
    pars0 = Truth
    correct_mcmc_samples = mcmc(pars0, ln_correct_posterior, 2 ** 16, (data, ))
    accept = correct_mcmc_samples[1:] == correct_mcmc_samples[:-1]
    print("acceptance ratio", np.mean(accept))
    print(correct_mcmc_samples)

    print("main: plotting correct posterior samples")
    labels = ["mean", "var"]
    ranges = [prior_info[0:2], prior_info[2:4]]
    fig = corner(correct_mcmc_samples, bins=128, labels=labels, range=ranges)
    pfn = "./correct_{}.png".format(Nstr)
    fig.savefig(pfn)
    print(pfn)

    print("main: computing meanvar and varvar")
    M = 2 ** 16
    print(N, M)
    means = np.zeros(M)
    vars = np.zeros(M)
    for m in range(M):
        foo = stats[0] + np.sqrt(stats[1]) * np.random.normal(size=N)
        em = np.mean(foo)
        ev = np.sum((foo - em) ** 2) / (N - 1)
        means[m] = em
        vars[m] = ev
    variances = np.array([np.var(means), np.var(vars)])
    print(variances)

    print("main: running pseudo MCMC")
    pars0 = Truth
    pseudo_mcmc_samples = mcmc(pars0, ln_pseudo_posterior, 2 ** 16, (stats, variances, ))
    accept = pseudo_mcmc_samples[1:] == pseudo_mcmc_samples[:-1]
    print("acceptance ratio", np.mean(accept))
    print(pseudo_mcmc_samples)

    print("main: plotting pseudo posterior samples")
    labels = ["mean", "var"]
    ranges = [prior_info[0:2], prior_info[2:4]]
    fig = corner(pseudo_mcmc_samples, bins=128, labels=labels, range=ranges)
    pfn = "./pseudo_{}.png".format(Nstr)
    fig.savefig(pfn)
    print(pfn)

if __name__ == "__main__":

    # setting TRUTHs
    Truth = np.array([7., 17.]) # mean, variance
    prior_info = np.array([0., 10., 0., 100.]) # min and max of mean and variance priors
    stepsizes = np.array([2., 2.])

    for N in [5, 23, 87, 167]:
        main(N)
