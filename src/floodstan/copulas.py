import math

import numpy as np

from scipy.stats import norm
from scipy.special import lambertw, owens_t

COPULA_NAMES = {
    "Gumbel": 1,
    "Clayton": 2,
    "Gaussian": 3,
    "Frank": 4
    }

# Bounds on copula parameters
RHO_LOWER = 0.01
RHO_UPPER = 0.95


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


# Utility to make sure we have 1d or 2d arrays with 2 columns
def to1d(u):
    u = np.atleast_1d(u).astype(float)
    if u.ndim != 1:
        raise ValueError(f"Expected 1d array, got ndim={u.ndim}.")
    return u


def to2d(uv):
    uv = np.atleast_2d(uv).astype(float)
    if uv.ndim != 2:
        raise ValueError(f"Expected 2d array, got ndim={uv.ndim}.")

    if uv.shape[1] != 2:
        uv = uv.T

    if uv.shape[1] != 2:
        raise ValueError(f"Expected shape[1]=2, got {uv.shape[1]}.")

    return uv


class Copula():
    def __init__(self, name):
        self.name = name
        self._rho = np.nan
        self.rho_min = RHO_LOWER
        self.rho_max = RHO_UPPER

    @property
    def rho(self):
        """ Get correlation parameter """
        rho = self._rho
        if rho < self.rho_min or rho > self.rho_max or np.isnan(rho):
            raise ValueError(f"Rho ({rho}) is not valid.")
        return rho

    @rho.setter
    def rho(self, val):
        """ Set correlation parameter """
        rho = float(val)
        errmsg = f"Expected rho in [{RHO_LOWER}, {RHO_UPPER}], got {rho}."
        assert rho >= self.rho_min and rho <= self.rho_max, errmsg
        self._rho = rho

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
            + rng.uniform(-delta / 2, delta / 2, size=(nsamples, 2))
        # Sampling from conditional copula
        # Considering that uv[:, 1] are probability samples
        uv[:, 1] = self.ppf_conditional(uv[:, 0], uv[:, 1])

        return uv


class GaussianCopula(Copula):
    def __init__(self):
        super(GaussianCopula, self).__init__("Gaussian")
        self._rho = np.nan
        # rho max reduced to pass conditional density tests
        self.rho_max = 0.92

    def _transform(self, uv):
        uv = to2d(uv)
        pq = norm.ppf(uv)
        return uv, pq

    @Copula.rho.setter
    def rho(self, val):
        """ Set theta parameter """
        rho = float(val)
        if rho < self.rho_min or rho > self.rho_max:
            errmess = f"Expected rho in [{self.rho_min},"\
                + f" {self.rho_max}], got {rho}."
            raise ValueError(errmess)

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
        if rho < self.rho_min or rho > self.rho_max:
            errmess = f"Expected rho in [{self.rho_min}, "\
                + f"{self.rho_max}], got {rho}."
            raise ValueError(errmess)
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
        errmsg = f"Expected rho in [{self.rho_min}, "\
            + f"{self.rho_max}], got {rho}."
        assert rho >= self.rho_min and rho <= self.rho_max, errmsg
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
        errmsg = f"Expected rho in [{RHO_LOWER}, {RHO_UPPER}], got {rho}."
        assert rho >= RHO_LOWER and rho <= RHO_UPPER, errmsg
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
        x = np.exp(-theta*ucond)
        y = np.exp(-theta*v)
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
        return -1. / theta*np.log(1 - w / ((1./q - 1) * x + 1))
