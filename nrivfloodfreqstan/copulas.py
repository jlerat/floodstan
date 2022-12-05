import math

import numpy as np
import pandas as pd

from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal
from scipy.special import lambertw

import pytest

RHO_MIN = 0.001
RHO_MAX = 0.999

def factory(name):
    if name == "Gaussian":
        return GaussianCopula()
    elif name == "Gumbel":
        return GumbelCopula()
    elif name == "Clayton":
        return ClaytonCopula()
    elif name == "Frank":
        return FrankCopula()
    else:
        raise ValueError(f"Cannot find copula {name}")


# Utility to make sure we have 1d or 2d arrays with 2 columns
def to1d(u):
    u = np.atleast_1d(u).astype(float)
    if u.ndim!=1:
        raise ValueError(f"Expected 1d array, got ndim={u.ndim}.")
    return u

def to2d(uv):
    uv = np.atleast_2d(uv).astype(float)
    if uv.ndim!=2:
        raise ValueError(f"Expected 2d array, got ndim={uv.ndim}.")

    if uv.shape[1]!=2:
        uv = uv.T

    if uv.shape[1]!=2:
        raise ValueError(f"Expected shape[1]=2, got {uv.shape[1]}.")

    return uv



# Copula base class
class Copula():
    def __init__(self, name):
        self.name = name
        self._rho = np.nan

    @property
    def rho(self):
        """ Get correlation parameter """
        rho = self._rho
        if rho<RHO_MIN or rho>RHO_MAX or np.isnan(rho):
            raise ValueError(f"Rho ({rho}) is not valid.")
        return rho

    @rho.setter
    def rho(self, val):
        """ Set correlation parameter """
        rho = float(val)
        errmsg = f"Expected rho in [{RHO_MIN}, {RHO_MAX}], got {rho}."
        assert rho>=RHO_MIN and rho<=RHO_MAX, errmsg
        self._rho = rho


    def pdf(self, uv):
        return NotImplementedError()

    def logpdf(self, uv):
        return np.log(self.pdf(uv))

    def pdf_ucensored(self, ucensor, v):
        return NotImplementedError()

    def logpdf_ucensored(self, ucensor, v):
        return np.log(self.logpdf_ucensored(ucensor, v))

    def cdf(self, uv):
        return NotImplementedError()

    def logcdf(self, uv):
        return np.log(self.cdf(uv))

    def ppf_conditional(self, ucond, b):
        return NotImplementedError()

    def sample(self, nsamples):
        # Latin hypercube sampling of uniform distributions
        delta = 1/nsamples/2
        uu = np.linspace(delta, 1-delta, nsamples)
        k1 = np.random.permutation(nsamples)
        k2 = np.random.permutation(nsamples)
        uv = np.column_stack([uu[k1], uu[k2]])+np.random.uniform(-delta/2, delta/2, \
                                        size=(nsamples, 2))

        # Sampling from conditional copula
        # Considering that uv[:, 1] are probability samples
        uv[:, 1] = self.ppf_conditional(uv[:, 0], uv[:, 1])

        return uv



class GaussianCopula(Copula):
    def __init__(self, approx=True):
        super(GaussianCopula, self).__init__("Gaussian")
        self.approx = approx

    def _transform(self, uv):
        uv = to2d(uv)
        pq = norm.ppf(uv)
        return uv, pq


    @Copula.rho.setter
    def rho(self, val):
        """ Set theta parameter """
        rho = float(val)
        errmsg = f"Expected rho in [{RHO_MIN}, {RHO_MAX}], got {rho}."
        assert rho>=RHO_MIN and rho<=RHO_MAX, errmsg
        self._rho = rho
        # Kendal Tau of Gaussian copula. See Joe (2014) Page 164.
        self.theta = math.sin(math.pi*rho/2)


    def pdf(self, uv):
        uv, pq = self._transform(uv)
        mu = np.zeros(2)
        theta = self.theta
        Sigma = np.array([[1, theta], [theta, 1]])
        # See Jones (2014), eq. 4.3 page 163
        A = norm.pdf(pq).prod(axis=1)
        return multivariate_normal.pdf(pq, mean=mu, cov=Sigma)/A


    def pdf_ucensored(self, ucensor, v):
        pcensor = norm.ppf(ucensor)
        q = norm.ppf(v)
        theta = self.theta
        sqr = math.sqrt(1-theta*theta)
        return norm.cdf((pcensor-theta*q)/sqr)


    def cdf(self, uv):
        uv, pq = self._transform(uv)

        if self.approx:
            # -- Alternative fast approx from
            # Zvi Drezner & G. O. Wesolowsky (1990) On the computation of the
            # bivariate normal integral,
            # Journal of Statistical Computation and Simulation, 35:1-2, 101-107, DOI: 10.1080/00949659008811236
            # https://www.tandfonline.com/doi/pdf/10.1080/00949659008811236?needAccess=true
            X = np.array([0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992])
            W = np.array([0.018854042, 0.038088059, 0.0452707394, 0.038088059,0.018854042])

            H1, H2 = pq.T
            R = self.theta
            BV = np.zeros_like(H1)

            H3 = H1*H2
            H12 = (H1*H1+H2*H2)/2.
            for i in range(len(X)):
                R1 = R*X[i]
                RR2 = 1.-R1*R1
                BV += W[i]*np.exp((R1*H3 - H12)/RR2)/math.sqrt(RR2)

            return BV*R+uv.prod(axis=1)

            # Translation of Drezner fortran code for case where rho>0.7
            #else:
            #    R2 = 1.-R*R
            #    R3 = math.sqrt(R2)
            #    H2 *= np.sign(R)
            #    H3 = H1*H2
            #    H7 = np.exp(-H3/2.)
            #    if R2>1e-10:
            #        H6 = np.abs(H1- H2)
            #        H5 = H6*H6/2.
            #        H6 = H6/R3
            #        AA = 0.5-H3/8.
            #        AB = 3.-2.*AA*H5
            #        BV = .13298076*H6*AB*norm.cdf(H6)-np.exp(-H5/R2)*(AB+AA*R2)*0.053051647
            #        for i in range(5):
            #            R1 = R3*X[i]
            #            RR = R1*R1
            #            R2 = math.sqrt(1.-RR)
            #            BV += -W[i]*np.exp(-H5/RR)*(np.exp(-H3/(1.+R2))/R2/H7-1.- AA*RR)

            #    if R>0:
            #        BV = BV*R3*H7+norm.cdf(np.maximum(H1, H2))
            #    else:
            #        BV = np.maximum(0, norm.cdf(H1)-norm.cdf(H2))-BV*R3*H7

            #    return BV

        else:
            # -- Very accurate method using scipy mvn --
            lower = np.zeros(2) # Does not matter here
            infin = np.zeros(2)
            correl = np.array([self.rho])
            nval = len(uv)
            csp = np.zeros(nval)
            for i in range(nval):
                upper = pq[[i]]
                err, csp[i], info = mvn.mvndst(lower, upper, infin, correl)

            return csp


    def ppf_conditional(self, ucond, b):
        pcond = norm.ppf(ucond)
        theta = self.theta
        return norm.cdf(theta*pcond+norm.ppf(b)*math.sqrt(1-theta*theta))


class GumbelCopula(Copula):
    def __init__(self):
        super(GumbelCopula, self).__init__("Gumbel")
        self.theta = np.nan

    @Copula.rho.setter
    def rho(self, val):
        """ Set theta parameter """
        rho = float(val)
        errmsg = f"Expected rho in [{RHO_MIN}, {RHO_MAX}], got {rho}."
        assert rho>=RHO_MIN and rho<=RHO_MAX, errmsg
        self._rho = rho
        self.theta = 1/(1-rho)


    def pdf(self, uv):
        uv = to2d(uv)
        xy = -np.log(uv)
        theta = self.theta
        expsum = np.power(xy, theta).sum(axis=1)
        exppow = np.power(expsum, 1./theta)
        F = np.exp(-exppow)

        return F*(exppow+theta-1)*np.power(expsum, 1/theta-2)\
                    *np.power(xy.prod(axis=1), theta-1)\
                    *1/uv.prod(axis=1)


    def pdf_ucensored(self, ucensor, v):
        raise NotImplementedError()


    def cdf(self, uv):
        uv = to2d(uv)
        xy = -np.log(uv)
        theta = self.theta
        expsum = np.power(xy, theta).sum(axis=1)
        return np.exp(-np.power(expsum, 1./theta))


    def sample(self, nsamples):
        # Sample first variable
        delta = 1/nsamples/2
        u = np.linspace(delta, 1-delta, nsamples)
        k1 = np.random.permutation(nsamples)
        p = u[k1]+np.random.uniform(-delta/2, delta/2, size=nsamples)

        # From cdf to exponential reduced var
        expu = -np.log(p)

        # Sample second variable
        k2 = np.random.permutation(nsamples)
        q = u[k2]+np.random.uniform(-delta/2, delta/2, size=nsamples)

        # solve for z in z+(m-1)*log(z) = x+(m-1)*log(x)+log(q)
        # See joe (2014), equation 4.15 page 172
        m = self.theta
        A = expu+(m-1)*np.log(expu)-np.log(q)
        B = A/(m-1)-math.log(m-1)

        # Solve with analytical solution obtained from lambertW function
        z = lambertw(np.exp(B))*(m-1)

        # Convert from z to exponential of reduced var 2
        expv = np.power(np.power(z, m)-np.power(expu, m), 1/m)

        return np.column_stack([p, np.exp(-expv)])



class ClaytonCopula(Copula):
    def __init__(self):
        super(ClaytonCopula, self).__init__("Clayton")
        self.theta = np.nan

    @Copula.rho.setter
    def rho(self, val):
        """ Set theta parameter """
        rho = float(val)
        errmsg = f"Expected rho in [{RHO_MIN}, {RHO_MAX}], got {rho}."
        assert rho>=RHO_MIN and rho<=RHO_MAX, errmsg
        self._rho = rho
        self.theta = 2*rho/(1-rho)


    def pdf(self, uv):
        uv = to2d(uv)
        theta = self.theta
        expsum = np.power(uv, -theta).sum(axis=1)-1
        return (1+theta)\
                    *np.power(uv.prod(axis=1), -theta-1) \
                    *np.power(expsum, -2-1/theta)


    def pdf_ucensored(self, ucensor, v):
        pass


    def cdf(self, uv):
        uv = to2d(uv)
        theta = self.theta
        expsum = np.power(uv, -theta).sum(axis=1)-1
        return np.power(expsum, -1./theta)


    def ppf_conditional(self, ucond, b):
        theta = self.theta
        # see Joe (2014), eq 4.10 page 168
        bb = np.power(b, -theta/(1+theta))-1
        cc = bb*np.power(ucond, -theta)+1
        return np.power(cc, -1/theta)



class FrankCopula(Copula):
    def __init__(self):
        super(FrankCopula, self).__init__("Frank")
        self.theta = np.nan

    @Copula.rho.setter
    def rho(self, val):
        """ Set theta parameter """
        rho = float(val)
        errmsg = f"Expected rho in [{RHO_MIN}, {RHO_MAX}], got {rho}."
        assert rho>=RHO_MIN and rho<=RHO_MAX, errmsg
        self._rho = rho
        self.theta = 2*rho/(1-rho)


    def pdf(self, uv):
        uv = to2d(uv)
        theta = self.theta
        x = np.exp(-theta*uv[:, 0])
        y = np.exp(-theta*uv[:, 1])
        w = 1-math.exp(-theta)
        z = w-(1-x)*(1-y)
        return theta*w*x*y/z/z


    def pdf_ucensored(self, ucensor, v):
        pass


    def cdf(self, uv):
        theta = self.theta
        x = np.exp(-theta*uv[:, 0])
        y = np.exp(-theta*uv[:, 1])
        w = 1-math.exp(-theta)
        z = w-(1-x)*(1-y)
        return -1/theta*np.log(z/w)


    def ppf_conditional(self, ucond, b):
        theta = self.theta
        # see Joe (2014), eq 4.7 page 165
        w = 1-math.exp(-theta)
        x = np.exp(-theta*ucond)
        return -1/theta*np.log(1-w/((1/b-1)*x+1))

