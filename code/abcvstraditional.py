"""
This file is part of the InferenceOfVariance project.
Copyright 2016 David W. Hogg (SCDA).

# To-do
- Doesn't conform to what's written in paper for Sigma^2
- Needs to output LaTeX table for inclusion in paper.
- Write M-H MCMC code. Right now it uses simple prior sampling, which SUX.
- Horrible, horrible `for` loops.
"""

import numpy as np
from corner import corner

def make_fake_data(N):
    return hoggdraw(Truth, N)

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

def hoggdraw(stats, N):
    return stats[0] + np.sqrt(stats[1]) * np.random.normal(size=N)

def hoggvar(data, mean):
    return np.sum((data - mean) ** 2) / (len(data) - 1)

def main(N, prior_samples=None):
    Nstr = "{:04d}".format(N)

    print("main: making fake data")
    np.random.seed(23)
    data = make_fake_data(N)
    empiricalmean = np.mean(data)
    empiricalvar = hoggvar(data, empiricalmean)
    stats = np.array([empiricalmean, empiricalvar])
    print(data, stats)

    print("main: running correct MCMC")
    pars0 = Truth
    correct_mcmc_samples = mcmc(pars0, ln_correct_posterior, 2 ** 16, (data, ))
    accept = correct_mcmc_samples[1:] == correct_mcmc_samples[:-1]
    print("acceptance ratio", np.mean(accept))
    thinfactor = 8
    correct_mcmc_samples = correct_mcmc_samples[::thinfactor] # thin
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
        foo = hoggdraw(stats, N)
        em = np.mean(foo)
        ev = hoggvar(foo, em)
        means[m] = em
        vars[m] = ev
    variances = np.array([np.var(means), np.var(vars)])
    print(variances)

    print("main: running pseudo MCMC")
    pars0 = Truth
    pseudo_mcmc_samples = mcmc(pars0, ln_pseudo_posterior, 2 ** 16, (stats, variances, ))
    accept = pseudo_mcmc_samples[1:] == pseudo_mcmc_samples[:-1]
    print("acceptance ratio", np.mean(accept))
    pseudo_mcmc_samples = pseudo_mcmc_samples[::thinfactor] # thin
    print(pseudo_mcmc_samples.shape, pseudo_mcmc_samples)

    print("main: plotting pseudo posterior samples")
    fig = corner(pseudo_mcmc_samples, bins=128, labels=labels, range=ranges)
    pfn = "./pseudo_{}.png".format(Nstr)
    fig.savefig(pfn)
    print(pfn)

    if prior_samples is None:
        print("main: getting prior samples (this might take a while)")
        pars0 = Truth
        prior_samples = mcmc(pars0, ln_prior, 2 ** 23, ())
        accept = prior_samples[1:] == prior_samples[:-1]
        print("acceptance ratio", np.mean(accept))
        print(prior_samples.shape, prior_samples)

    print("main: running ABC censoring (this might take a while)")
    squared_distances = np.zeros(len(prior_samples))
    for i, pars in enumerate(prior_samples):
        data = hoggdraw(pars, N)
        mean = np.mean(data)
        squared_distances[i] = -2. * ln_pseudo_posterior([mean, hoggvar(data, mean)], stats, variances)
    percentile = 100. * len(correct_mcmc_samples) / len(prior_samples)
    threshold = np.percentile(squared_distances, percentile)
    abc_samples = prior_samples[squared_distances < threshold]
    print("threshold", threshold)
    print(abc_samples.shape, abc_samples)

    print("main: plotting ABC samples")
    fig = corner(abc_samples, bins=128, labels=labels, range=ranges)
    pfn = "./abc_{}.png".format(Nstr)
    fig.savefig(pfn)
    print(pfn)

    return prior_samples

if __name__ == "__main__":

    # setting TRUTHs
    Truth = np.array([7., 17.]) # mean, variance
    prior_info = np.array([0., 10., 0., 100.]) # min and max of mean and variance priors
    stepsizes = np.array([2., 2.])

    prior_samples = None
    for N in [5, ]:
        prior_samples = main(N, prior_samples=prior_samples)
