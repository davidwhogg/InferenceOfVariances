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

def ln_correct_likelihood(data, pars):
    mean, var = pars
    return np.sum(-0.5 * (data - mean) ** 2 / var) - (0.5 * len(data) * np.log(var))

def ln_correct_posterior(pars, data):
    lnp = ln_prior(pars)
    if not np.isfinite(lnp):
        return -np.Inf
    return lnp + ln_correct_likelihood(data, pars)

def ln_pseudo_likelihood(empiricalvar, pars, varvar):
    mean, var = pars
    return (-0.5 * (empiricalvar - var ) ** 2 / varvar)

def ln_pseudo_posterior(pars, empiricalvar, varvar):
    lnp = ln_prior(pars)
    if not np.isfinite(lnp):
        return -np.Inf
    return lnp + ln_pseudo_likelihood(empiricalvar, pars, varvar)

def ln_abc_function(pars, empiricalvar, varvar, N, threshold):
    lnp = ln_prior(pars)
    if not np.isfinite(lnp):
        return -np.Inf
    mean, var = pars
    thisdata = mean + np.sqrt(var) * np.random.normal(size=N)
    thispars = [mean, hoggvar(thisdata)]
    if ln_pseudo_likelihood(empiricalvar, thispars, varvar) < -0.5 * threshold: # NEGATIVE 0.5
        return -np.Inf
    return lnp

def mcmc_step(pars, lnpvalue, lnp, args):
    newpars = pars + stepsizes * np.random.normal(size=len(pars))
    newlnpvalue = lnp(newpars, *args)
    if (newlnpvalue - lnpvalue) > np.log(np.random.uniform()):
        return newpars, newlnpvalue
    return pars.copy(), lnpvalue

def mcmc(pars0, lnp, nsteps, args, abc=False):
    pars = pars0.copy()
    if abc:
        lnpvalue = ln_prior(pars)
    else: 
        lnpvalue = lnp(pars, *args)
    parss = np.zeros((nsteps, len(pars)))
    for k in range(nsteps):
        pars, lnpvalue = mcmc_step(pars, lnpvalue, lnp, args)
        parss[k,:] = pars
    return parss

def hoggdraw(stats, N):
    return stats[0] + np.sqrt(stats[1]) * np.random.normal(size=N)

def hoggvar(data):
    mean = np.mean(data)
    return np.sum((data - mean) ** 2) / float(len(data) - 1)

def main(N):
    np.random.seed(23)
    Nstr = "{:04d}".format(N)

    print("main: making fake data")
    data = make_fake_data(N)
    empiricalmean = np.mean(data)
    empiricalvar = hoggvar(data)
    stats = np.array([empiricalmean, empiricalvar])
    print(data, stats)

    print("main: making LaTeX input file")
    lfn = "./data_{}.tex".format(Nstr)
    fd = open(lfn, "w")
    fd.write("% this file was created by abc.py\n")
    fd.write(r"\newcommand{\samples}{")
    for n in range(N):
        fd.write("{:.2f}".format(data[n]))
        if n < (N - 1):
            fd.write(r", ")
    fd.write("}\n")
    fd.write(r"\newcommand{\samplemean}{")
    fd.write("{:.2f}".format(empiricalmean))
    fd.write("}\n")
    fd.write(r"\newcommand{\samplevar}{")
    fd.write("{:.2f}".format(empiricalvar))
    fd.write("}\n")
    fd.close()
    print(lfn)

    # MCMC sampling and plotting shared parameters
    pars0 = Truth.copy()
    Tbig = 2 ** 19
    thinfactor = 2 ** 4
    varvar = 2. * empiricalvar * empiricalvar / float(N - 1)
    bins = 32
    labels = [r"mean $\mu$", r"variance $V$"]
    ranges = [prior_info[0:2], prior_info[2:4]]

    print("main: running ABC (this might take a while)")
    for log2thresh in [2, 0, -2. -4, ]:
        thresh = 2. ** log2thresh
        print("working on threshold", thresh)
        abc_mcmc_samples = mcmc(pars0, ln_abc_function, 2*Tbig, (empiricalvar, varvar, N, thresh), abc=True)
        accept = abc_mcmc_samples[1:] != abc_mcmc_samples[:-1]
        print("acceptance ratio", np.mean(accept))
        abc_mcmc_samples = abc_mcmc_samples[::thinfactor*2] # thin
        print(abc_mcmc_samples.shape)

        print("main: plotting ABC posterior samples")
        fig = corner(abc_mcmc_samples, bins=bins, labels=labels, range=ranges)
        pfn = "./abc_{}_{}.png".format(Nstr, log2thresh)
        fig.savefig(pfn)
        print(pfn)

    print("main: running correct MCMC")
    correct_mcmc_samples = mcmc(pars0, ln_correct_posterior, Tbig, (data, ))
    accept = correct_mcmc_samples[1:] != correct_mcmc_samples[:-1]
    print("acceptance ratio", np.mean(accept))
    correct_mcmc_samples = correct_mcmc_samples[::thinfactor] # thin
    print(correct_mcmc_samples.shape)

    print("main: plotting correct posterior samples")
    fig = corner(correct_mcmc_samples, bins=bins, labels=labels, range=ranges)
    pfn = "./correct_{}.png".format(Nstr)
    fig.savefig(pfn)
    print(pfn)

    print("main: running pseudo MCMC")
    pseudo_mcmc_samples = mcmc(pars0, ln_pseudo_posterior, Tbig, (empiricalvar, varvar, ))
    accept = pseudo_mcmc_samples[1:] != pseudo_mcmc_samples[:-1]
    print("acceptance ratio", np.mean(accept))
    pseudo_mcmc_samples = pseudo_mcmc_samples[::thinfactor] # thin
    print(pseudo_mcmc_samples.shape)

    print("main: plotting pseudo posterior samples")
    fig = corner(pseudo_mcmc_samples, bins=bins, labels=labels, range=ranges)
    pfn = "./pseudo_{}.png".format(Nstr)
    fig.savefig(pfn)
    print(pfn)

if __name__ == "__main__":

    # setting TRUTHs
    Truth = np.array([7., 17.]) # mean, variance
    prior_info = np.array([0., 10., 0., 100.]) # min and max of mean and variance priors
    stepsizes = np.array([4., 4.])

    N = 5
    main(N)
