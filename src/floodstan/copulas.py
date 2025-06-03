import math

import numpy as np

from scipy.stats import norm
from scipy.special import lambertw, owens_t
from floodstan.marginals import TruncatedNormalParameterPrior
from floodstan.data_processing import to2d

COPULA_NAMES = {
    "Gumbel": 1,
    "Clayton": 2,
    "Gaussian": 3,
    "Frank": 4
    }

# Bounds on copula parameters
RHO_LOWER = 0.01
RHO_UPPER = 0.95

# Prior on copula parameter
RHO_PRIOR_LOC_DEFAULT = 0.7
RHO_PRIOR_SCALE_DEFAULT = 1.


def factory(name):
    txt = "/".join(COPULA_NAMES.keys())
    if name not in COPULA_NAMES:
        errmess = f"Expected copula name in {txt}, got {name}."
        raise ValueError(errmess)

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


class Copula():
    def __init__(self, name):
        self.name = name
        self._rho = np.nan
        self.rho_lower = RHO_LOWER
        self.rho_upper = RHO_UPPER

    @property
    def rho(self):
        """ Get correlation parameter """
        rho = self._rho
        self._check_rho(rho)
        return rho

    @rho.setter
    def rho(self, val):
        """ Set correlation parameter """
        rho = float(val)
        self._check_rho(rho)
        self._rho = rho

    @property
    def rho_prior(self):
        prior = TruncatedNormalParameterPrior("rho")
        prior._lower = self.rho_lower
        prior._upper = self.rho_upper
        prior._loc = RHO_PRIOR_LOC_DEFAULT
        prior._scale = RHO_PRIOR_SCALE_DEFAULT
        prior.uninformative = True
        return prior

    def _check_rho(self, rho):
        if rho < self.rho_lower or rho > self.rho_upper:
            errmsg = f"Expected rho in [{self.rho_lower}, "\
                     + f"{self.rho_upper}], got {rho}."
            raise ValueError(errmsg)

    def pdf(self, uv):
        return NotImplementedError()

    def logpdf(self, uv):
        return np.log(self.pdf(uv))

    def conditional_density(self, ucond, v):
        return NotImplementedError()

    def cdf(self, uv):
        return NotImplementedError()

    def logcdf(self, uv):
        return np.log(self.cdf(uv))

    def ppf_conditional(self, ucond, q):
        return NotImplementedError()

    def sample(self, nsamples, seed=5446):
        # Latin hypercube sampling of uniform distributions
        delta = 1. / nsamples / 2
        uu = np.linspace(delta, 1 - delta, nsamples)
        rng = np.random.default_rng(seed)
        k1 = rng.permutation(nsamples)
        k2 = rng.permutation(nsamples)
        uv = np.column_stack([uu[k1], uu[k2]])\
            + np.random.uniform(-delta / 2, delta / 2, size=(nsamples, 2))
        # Sampling from conditional copula
        # Considering that uv[:, 1] are probability samples
        uv[:, 1] = self.ppf_conditional(uv[:, 0], uv[:, 1])

        return uv

    def neglogpost(self, theta, data, icases,
                   marginaly, marginalz,
                   ycensor, zcensor):
        # Get params
        yparams = theta[:3]
        zparams = theta[3:-1]

        rho = theta[-1]

        if rho < self.rho_lower or rho > self.rho_upper:
            return -np.inf
        self.rho = rho

        try:
            marginaly.params = yparams
        except Exception:
            return -np.inf

        try:
            marginalz.params = zparams
        except Exception:
            return -np.inf

        # Get cdfs and pdfs and censor cdfs
        cdfs, logpdfs = data * 0., data * 0.
        cdf_censors = [0, 0]

        ydata = data[:, 0]
        cdfs[:, 0] = marginaly.cdf(ydata)
        cdf_censors[0] = marginaly.cdf(ycensor)
        logpdfs[:, 0] = marginaly.logpdf(ydata)

        zdata = data[:, 1]
        cdfs[:, 1] = marginalz.cdf(zdata)
        cdf_censors[1] = marginalz.cdf(zcensor)
        logpdfs[:, 1] = marginalz.logpdf(zdata)

        # initialise with prior
        nlp = -self.rho_prior.logpdf(rho)

        nlp -= marginaly.locn_prior.logpdf(yparams[0])
        nlp -= marginaly.logscale_prior.logpdf(yparams[1])
        if marginaly.has_shape:
            nlp -= marginaly.shape1_prior.logpdf(yparams[2])

        nlp -= marginalz.locn_prior.logpdf(zparams[0])
        nlp -= marginalz.logscale_prior.logpdf(zparams[1])
        if marginalz.has_shape:
            nlp -= marginalz.shape1_prior.logpdf(zparams[2])

        # Cases with z observed
        i11 = icases.i11
        if i11.sum() > 0:
            # Marginal likelihood
            nlp -= logpdfs[i11].sum()
            # Copula likelihood
            nlp -= self.logpdf(cdfs[i11]).sum()

            if np.isnan(nlp):
                return -np.inf

        i21 = icases.i21
        if i21.sum() > 0:
            # Marginal likelihood for z
            nlp -= logpdfs[i21, 1].sum()
            # Conditional copula likelihood for ycensor
            nlp -= self.conditional_density(cdf_censors[0],
                                            cdfs[i21, 1]).sum()
            if np.isnan(nlp):
                return -np.inf

        i31 = icases.i31
        if i31.sum() > 0:
            # Marginal likelihood for z
            nlp -= logpdfs[i31, 1].sum()
            if np.isnan(nlp):
                return -np.inf

        # Cases with z censored
        i12 = icases.i12
        if i12.sum() > 0:
            # Marginal likelihood for y
            nlp -= logpdfs[i12, 0].sum()
            # Conditional copula likelihood for zcensor
            nlp -= self.conditional_density(cdf_censors[1],
                                            cdfs[i12, 0]).sum()
            if np.isnan(nlp):
                return -np.inf

        i22 = icases.i22
        n22 = i22.sum()
        if n22 > 0:
            # Copula likelihood for both censors
            nlp -= n22 * self.logpdf(cdf_censors)[0]
            if np.isnan(nlp):
                return -np.inf

        i32 = icases.i32
        n32 = i32.sum()
        if n32 > 0:
            # Censored likelihood for zcensor
            nlp -= n32 * math.log(cdf_censors[1])
            if np.isnan(nlp):
                return -np.inf

        # Cases with z missing
        i13 = icases.i13
        if i13.sum() > 0:
            # Marginal likelihood for y
            nlp -= logpdfs[i13, 0].sum()
            if np.isnan(nlp):
                return -np.inf

        i23 = icases.i23
        n23 = i23.sum()
        if n23 > 0:
            # Censored likelihood for ycensor
            nlp -= n23 * math.log(cdf_censors[0])
            if np.isnan(nlp):
                return -np.inf

        return nlp


class GaussianCopula(Copula):
    def __init__(self):
        super(GaussianCopula, self).__init__("Gaussian")
        self._rho = np.nan
        # rho max reduced to pass conditional density tests
        self.rho_upper = 0.92

        # rho max reduced to pass conditional density tests
        self.rho_upper = 0.92

    def _transform(self, uv):
        uv = to2d(uv)
        pq = norm.ppf(uv)
        return uv, pq

    @Copula.rho.setter
    def rho(self, val):
        """ Set theta parameter """
        rho = float(val)
        self._check_rho(rho)
        self._rho = rho
        # Kendal Tau of Gaussian copula. See Joe (2014) Page 164.
        self.theta = math.sin(math.pi * rho / 2)

    def pdf(self, uv):
        uv, pq = self._transform(uv)
        theta = self.theta
        # See Jones (2014), eq. 4.3 page 163
        s2 = (pq*pq).sum(axis=1)
        z = s2 - 2 * theta * pq.prod(axis=1)
        r2 = 1 - theta*theta
        return 1. / math.sqrt(r2) * np.exp(-z / r2 / 2. + s2 / 2.)

    def conditional_density(self, ucond, v):
        pcensor = norm.ppf(ucond)
        q = norm.ppf(v)
        theta = self.theta
        sqr = math.sqrt(1. - theta * theta)
        return norm.cdf((q - theta * pcensor) / sqr)

    def cdf(self, uv):
        uv, pq = self._transform(uv)

        r = self.theta
        z1 = norm.ppf(uv[:, 0])
        z2 = norm.ppf(uv[:, 1])

        denom = math.sqrt((1 + r) * (1 - r))
        a1 = (z2 / z1 - r) / denom
        a2 = (z1 / z2 - r) / denom
        product = z1 * z2
        delta = (product < 0) | ((product == 0) & (z1 + z2 < 0))

        return 0.5 * (uv[:, 0] + uv[:, 1] - delta)\
            - owens_t(z1, a1) - owens_t(z2, a2)

    def ppf_conditional(self, ucond, q):
        pcond = norm.ppf(ucond)
        theta = self.theta
        return norm.cdf(theta * pcond + norm.ppf(q)
                        * math.sqrt(1. - theta*theta))


class GumbelCopula(Copula):
    def __init__(self):
        super(GumbelCopula, self).__init__("Gumbel")
        self.theta = np.nan

    @Copula.rho.setter
    def rho(self, val):
        """ Set theta parameter """
        rho = float(val)
        self._check_rho(rho)
        self._rho = rho
        self.theta = 1. / (1. - rho)

    def pdf(self, uv):
        uv = to2d(uv)
        xy = -np.log(uv)
        theta = self.theta
        expsum = np.power(xy, theta).sum(axis=1)
        exppow = np.power(expsum, 1. / theta)
        F = np.exp(-exppow)
        return F * (exppow + theta - 1.) * np.power(expsum, 1. / theta - 2.)\
            * np.power(xy.prod(axis=1), theta - 1.)\
            * 1. / uv.prod(axis=1)

    def conditional_density(self, ucond, v):
        x = -np.log(ucond)
        y = -np.log(v)
        theta = self.theta
        e1 = np.exp(-np.power(np.power(x, theta)
                              + np.power(y, theta), 1. / theta))
        e2 = np.power(1. + np.power(y / x, theta), 1. / theta - 1.)
        return 1. / ucond * e1 * e2

    def cdf(self, uv):
        uv = to2d(uv)
        xy = -np.log(uv)
        theta = self.theta
        expsum = np.power(xy, theta).sum(axis=1)
        return np.exp(-np.power(expsum, 1. / theta))

    def sample(self, nsamples):
        # Sample first variable
        delta = 1. / nsamples / 2.
        u = np.linspace(delta, 1. - delta, nsamples)
        k1 = np.random.permutation(nsamples)
        p = u[k1] + np.random.uniform(-delta / 2., delta / 2., size=nsamples)

        # From cdf to exponential reduced var
        expu = -np.log(p)

        # Sample second variable
        k2 = np.random.permutation(nsamples)
        q = u[k2] + np.random.uniform(-delta / 2., delta / 2., size=nsamples)

        # solve for z in z+(m-1)*log(z) = x+(m-1)*log(x)+log(q)
        # See joe (2014), equation 4.15 page 172
        m = self.theta
        A = expu + (m - 1)*np.log(expu) - np.log(q)
        B = A / (m - 1) - math.log(m - 1)

        # Solve with analytical solution obtained from lambertW function
        # Using asymptotic expansion of LambertW for x->inf:
        # LambertW(x) ~ ln(x)-ln(ln(x))
        z = np.where(B < 700, lambertw(np.exp(B)), B - np.log(B)) * (m - 1)

        # Convert from z to exponential of reduced var
        expv = np.power(np.power(z, m)-np.power(expu, m), 1. / m)

        # Return non-reduced var
        return np.column_stack([p, np.exp(-expv).real])


class ClaytonCopula(Copula):
    def __init__(self):
        super(ClaytonCopula, self).__init__("Clayton")
        self.theta = np.nan

    @Copula.rho.setter
    def rho(self, val):
        """ Set theta parameter """
        rho = float(val)
        self._check_rho(rho)
        self._rho = rho
        self.theta = 2. * rho/(1. - rho)

    def pdf(self, uv):
        uv = to2d(uv)
        theta = self.theta
        expsum = np.power(uv, -theta).sum(axis=1) - 1.
        return (1 + theta)\
            * np.power(uv.prod(axis=1), -theta - 1.) \
            * np.power(expsum, -2. - 1. / theta)

    def conditional_density(self, ucond, v):
        theta = self.theta
        # see Joe (2014), eq 4.10 page 168
        return np.power(1 + np.power(ucond, theta)
                        * (np.power(v, -theta) - 1.), -1. - 1. / theta)

    def cdf(self, uv):
        uv = to2d(uv)
        theta = self.theta
        expsum = np.power(uv, -theta).sum(axis=1) - 1.
        return np.power(expsum, -1. / theta)

    def ppf_conditional(self, ucond, q):
        theta = self.theta
        # see Joe (2014), eq 4.10 page 168
        bb = np.power(q, -theta / (1. + theta)) - 1.
        cc = bb*np.power(ucond, -theta) + 1.
        return np.power(cc, -1. / theta)


class FrankCopula(Copula):
    def __init__(self):
        super(FrankCopula, self).__init__("Frank")
        self.theta = np.nan

    @Copula.rho.setter
    def rho(self, val):
        """ Set theta parameter """
        rho = float(val)
        self._check_rho(rho)
        self._rho = rho
        self.theta = 2. * rho / (1. - rho)

    def pdf(self, uv):
        uv = to2d(uv)
        theta = self.theta
        x = np.exp(-theta * uv[:, 0])
        y = np.exp(-theta * uv[:, 1])
        w = 1. - math.exp(-theta)
        z = w - (1. - x) * (1 - y)
        return theta * w * x * y / z / z

    def conditional_density(self, ucond, v):
        theta = self.theta
        # see Joe (2014), eq 4.7 page 165
        w = 1 - math.exp(-theta)
        x = np.exp(-theta * ucond)
        y = np.exp(-theta * v)
        return x / (w / (1 - y) - (1-x))

    def cdf(self, uv):
        theta = self.theta
        x = np.exp(-theta * uv[:, 0])
        y = np.exp(-theta * uv[:, 1])
        w = 1 - math.exp(-theta)
        z = w - (1 - x) * (1 - y)
        return -1/theta*np.log(z/w)

    def ppf_conditional(self, ucond, q):
        theta = self.theta
        # see Joe (2014), eq 4.7 page 165
        w = 1 - math.exp(-theta)
        x = np.exp(-theta * ucond)
        return -1. / theta * np.log(1 - w / ((1./q - 1) * x + 1))
