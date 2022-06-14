#!/usr/bin/env python3

import dadi
import dadi.Godambe
import numpy as np
from sys import exit

###uncertainty analysis for IM###
#script from https://github.com/pblischak/inbreeding-sfs/tree/e7b5ba4828617d65642bf2e7fbbfb32171c9d243/data

def IM(params, ns, pts):
    """
    Split into two populations, with different migration rates.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations) 
    m12: Migration from pop 2 to pop 1 (2*Na*m12)
    m21: Migration from pop 1 to pop 2
    """
    nu1, nu2, m12, m21, T = params

    xx = dadi.Numerics.default_grid(pts)
    
    phi = dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    
    phi = dadi.Integration.two_pops(phi, xx, T, nu1, nu2, m12=m12, m21=m21)
    fs = dadi.Spectrum.from_phi(phi, ns, (xx,xx))
    
    return fs   
    

if __name__ == '__main__':
    # Read in the bootstrapped spectra. DO NOT fold
    all_boot = [dadi.Spectrum.from_file('./ophionotus/analyses/dadi_cli/bs/vitoriae_hexactis_rmSGvix_BMhex.bootstrapping.bootstrapping.{}.fs'.format(i)) for i in range(100)]

    # Now read in the original data set
    data = dadi.Spectrum.from_file("./ophionotus/analyses/dadi/victoriae-hexactis.sfs")
    pts_l = [400, 410, 420]
    func = IM
    popt = [29.7679, 2.3432, 0.0158, 5.0511, 13.0969]

    func_ex = dadi.Numerics.make_extrap_log_func(func)

    # Calculate parameter uncertainties
    """
    We include this section to check uncertainties across different step sizes (eps).
    This can be turned off by setting check_grid_size=False.
    """
    check_grid_sizes = False
    if check_grid_sizes:
        print("\nChecking uncertainties with different grid sizes:")
        for e in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    	    u = dadi.Godambe.GIM_uncert(func_ex, pts_l, all_boot, popt, data, log=True, multinom=True, eps=e)
    	    print("{} = {}".format(e,u))
        exit(0)
    
    eps=0.1 # the default is 1e-2
    uncerts,GIM = dadi.Godambe.GIM_uncert(func_ex, pts_l, all_boot, popt, data, log=True, multinom=True, return_GIM=True, eps=eps)
    vcov = np.linalg.inv(GIM)

    # Set conversion parameters
    theta = 181.81 # estimated from model
    L = 1917004
    mu = 1.43e-8
    g  = 9 # generation time
    scalar = L*mu*4
    Nref = theta / scalar
    print("\n\nConverting parameters to actual units (eps={})...\n".format(eps))
    print("Using the following values for conversion:")
    print("  Sequence length = {}".format(L))
    print("  Mutation rate =   {}".format(mu))
    print("  Generation time = {}".format(g))
    print("")

    uncerts2 = [
        np.sqrt(vcov[-1,-1] + vcov[0,0] + 2*vcov[0,-1]), #nu1
        np.sqrt(vcov[-1,-1] + vcov[1,1] + 2*vcov[1,-1]), #nu2
        np.sqrt(vcov[-1,-1] + vcov[2,2] + 2*vcov[2,-1]), #m12
        np.sqrt(vcov[-1,-1] + vcov[3,3] + 2*vcov[3,-1]), #m21
        np.sqrt(vcov[-1,-1] + vcov[4,4] + 2*vcov[4,-1]), #T
        uncerts[-1]
    ]
    
    log_params = [
        np.log(theta) + np.log(popt[0]) + np.log(1/scalar),   # nu1
        np.log(theta) + np.log(popt[1]) + np.log(1/scalar),   # nu2
        np.log(theta) + np.log(popt[2]) + np.log(1/scalar),   # m12
        np.log(theta) + np.log(popt[3]) + np.log(1/scalar),   # m21
        np.log(theta) + np.log(popt[4]) + np.log(2*g/scalar),   # T
    ]
    
    print("Estimated parameter standard deviations from GIM:\n{}\n".format(uncerts))
    print("Estimated parameter standard deviations from error propagation:\n{}\n".format(uncerts2))
    print("Variance-Covariance Matrix:\n{}\n".format(vcov))

    """
    With propogation of uncertainty
    """
    print("\nParameter estimates and 95% confidence intervals:")
    print("Nref = {} ({}--{})".format(Nref, np.exp(np.log(theta)-1.96*uncerts2[-1])/scalar, np.exp(np.log(theta)+1.96*uncerts2[-1])/scalar))
    print("nu1   = {} ({}--{})".format(popt[0]*Nref, np.exp(log_params[0]-1.96*uncerts2[0]), np.exp(log_params[0]+1.96*uncerts2[0])))
    print("nu2   = {} ({}--{})".format(popt[1]*Nref, np.exp(log_params[1]-1.96*uncerts2[1]), np.exp(log_params[1]+1.96*uncerts2[1])))
    print("m12   = {} ({}--{})".format(popt[2]*Nref, np.exp(log_params[2]-1.96*uncerts2[2]), np.exp(log_params[2]+1.96*uncerts2[2])))
    print("m21   = {} ({}--{})".format(popt[3]*Nref, np.exp(log_params[3]-1.96*uncerts2[3]), np.exp(log_params[3]+1.96*uncerts2[3])))
    print("T   = {} ({}--{})".format(popt[4]*2*g*Nref, np.exp(log_params[4]-1.96*uncerts2[4]), np.exp(log_params[4]+1.96*uncerts2[4])))
