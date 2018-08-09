import numpy as np
from scipy import optimize


class SignalModel(object):
    """General signal class:
    model_size is just number of parameters"""

    def __init__(self, parameters, interval_length=1):
        self.parameters = np.array(parameters).astype(float)
        self.model_size = len(self.parameters)
        self.interval_length = interval_length

    def get_samples(self, positions):
        """
        Returns sample values for given positions
        Args:
            positions (numpy.ndarray): positions to evaluate

        Returns:
            numpy.ndarray: sample values
        """
        raise NotImplementedError('get_samples must be implemented by the subclass')

    def norm2(self, parameters=None):
        """
        Args:
            parameters (numpy.ndarray): optional, if provided function returns
                norm of the signal defined by parameters
        Returns:
            float: value of the L2 norm of the (continous) signal
        """
        raise NotImplementedError('norm2 must be implemented by the subclass')

    def path(self, point, change, n=50):
        raise NotImplementedError('path must be implemented by the subclass')

    def square_error(self, parameters2):
        """
        Returns the value of MSE, or squared difference between self and signal2,
        (treated as continuous functions).

        Args:
            parameters2 (numpy.ndarray):  signal to compare to

        Returns:
            float: value of MSE

        """
        diff_parameters = self.parameters - parameters2
        return self.norm2(diff_parameters)


class SignalPolynomial(SignalModel):
    """Polynomial signal on interval [0,1]"""

    def __init__(self, parameters, interval_length=1):
        super(SignalPolynomial, self).__init__(parameters, interval_length)

    def get_samples(self, positions):
        positions = np.array(positions)
        samples = np.zeros(np.shape(positions))
        for k in range(0, self.model_size):
            samples += self.parameters[-(k + 1)] * (positions ** k)
        return samples

    def norm2(self, prm=None):
        if prm is None:
            prm = self.parameters
        norm = 0
        for k in range(0, len(prm)):
            for m in range(0, len(prm)):
                norm += self.interval_length ** (k + m + 1) * prm[-(k + 1)] * prm[-(m + 1)] / (k + m + 1)
        return norm

    def path(self, start_pos, change, n=50):
        start_pos = np.array(start_pos)
        p = [start_pos]
        value = self.get_samples(start_pos)
        new_pol = np.copy(self.parameters)
        new_pol[-1] -= value
        for i in range(1, n):
            new_pol += change / (n - 1)
            r = np.roots(new_pol)
            p.append(min(r, key=lambda x: abs(x - p[i - 1])))
        return p

    @classmethod
    def create_ls_matrix(cls, sample_positions, model_size):
        x = np.zeros((len(sample_positions), model_size))
        for k in range(0, model_size):
            x[:, model_size - k - 1] = np.power(sample_positions, k)
        return x

    @classmethod
    def create_derivative_ls_matrix(cls, sample_positions, model_size):
        x = np.zeros((len(sample_positions), model_size))
        for k in range(1, model_size):
            x[:, model_size - k - 1] = k * np.power(sample_positions, k - 1)
        return x

    @classmethod
    def compute_ls_gradient(cls, positions, parameters, samples):
        x = cls.create_ls_matrix(positions, len(parameters))
        dx = cls.create_derivative_ls_matrix(positions, len(parameters))
        g = - 2 * np.dot(dx, parameters) * (samples - np.dot(x, parameters))
        return g

    @property
    def degree(self):
        return self.model_size - 1


class ConstrainedPolynomial(SignalPolynomial):
    """Abstract constrained polynomial:
    contains all the logic needed for ALS algorithm, but constrains need to be added"""

    def __init__(self, parameters, interval_length=1):
        super(ConstrainedPolynomial, self).__init__(parameters, interval_length)

    @classmethod
    def create_ls_matrix(cls, sample_positions, model_size, tr_param=0):
        sample_positions = cls.shifted_positions(sample_positions, tr_param)
        return super(ConstrainedPolynomial, cls).create_ls_matrix(sample_positions, model_size)

    @classmethod
    def create_derivative_ls_matrix(cls, sample_positions, model_size, tr_param=0):
        sample_positions = cls.shifted_positions(sample_positions, tr_param)
        return super(ConstrainedPolynomial, cls).create_derivative_ls_matrix(sample_positions, model_size)

    @classmethod
    def positions_derivative(cls, sample_positions, tr_parameter):
        raise NotImplementedError

    @staticmethod
    def shifted_positions(sample_positions, trace_param):
        raise NotImplementedError

    @staticmethod
    def zero_transformation():
        raise NotImplementedError

    @classmethod
    def compute_ls_gradient(cls, positions, parameters, samples, tr_param=0):
        x = cls.create_ls_matrix(positions, len(parameters), tr_param)
        dx = cls.create_derivative_ls_matrix(positions, len(parameters), tr_param)
        dtr = cls.positions_derivative(positions, tr_param)
        g = -2 * np.dot(
            (samples - np.dot(x, parameters)).T,
            np.dot(
                np.diag(np.dot(dx, parameters)),
                dtr))
        return g


class SurfacePolynomial(ConstrainedPolynomial):
    """Simple version of constrained polynomial on the surface:
    constrains modeled as simple rational function, x/(1-parameters*x)"""

    def __init__(self, parameters, interval_length=1):
        super(SurfacePolynomial, self).__init__(parameters, interval_length)

    @classmethod
    def positions_derivative(cls, sample_positions, tr_parameter):
        return np.array([x ** 2 for x in cls.shifted_positions(sample_positions, tr_parameter)])

    @staticmethod
    def shifted_positions(sample_positions, trace_param):
        assert np.abs(trace_param) <= 1
        return [x / (1 - trace_param * x) for x in sample_positions]

    @staticmethod
    def zero_transformation():
        return 0


class FullSurfacePolynomial(ConstrainedPolynomial):
    """Simple version of constrained polynomial on the surface:
    constrains modeled as simple rational function, parameters[1]*x/(1-parameters[0]*x)"""

    def __init__(self, parameters, interval_length=1):
        super(FullSurfacePolynomial, self).__init__(parameters, interval_length)

    @classmethod
    def positions_derivative(cls, sample_positions, tr_parameter):
        coef = 1.0 / (1.0 + tr_parameter[1])
        return np.array([[coef * x ** 2, coef * x]
                         for x in cls.shifted_positions(sample_positions, tr_parameter)])

    @staticmethod
    def shifted_positions(sample_positions, trace_param):
        assert np.abs(trace_param[0]) <= 1
        assert trace_param[1] > -1
        return [(1 + trace_param[1]) * x / (1 - trace_param[0] * x) for x in sample_positions]

    @staticmethod
    def zero_transformation():
        return [0, 0]


class SecondSurfacePolynomial(ConstrainedPolynomial):
    """The final version of constrained polynomial on the surface, as described in the paper"""

    def __init__(self, parameters, interval_length=1):
        super(SecondSurfacePolynomial, self).__init__(parameters, interval_length)

    @classmethod
    def positions_derivative(cls, sample_positions, tr_parameter):
        cos_a = np.cos(tr_parameter[0])
        sin_a = np.sin(tr_parameter[0])
        b = tr_parameter[1]
        f = tr_parameter[2]
        return np.array([[cls._denominator(s, f, cos_a, sin_a) ** 2 * ((f * sin_a) + (s * cos_a)) * (b * s),
                          s * cls._denominator(s, f, cos_a, sin_a), 0]
                         for s in sample_positions])

    @staticmethod
    def _denominator(s, f, cosa, sina):
        return 1.0 / (f * cosa - s * sina)

    @staticmethod
    def shifted_positions(sample_positions, trace_param):
        assert trace_param[1] > 0, 'b = ' + str(trace_param[1])
        assert abs(trace_param[0]) < (np.pi / 2.0), 'a = ' + str(trace_param[0])
        assert abs(np.tan(trace_param[0])) < trace_param[2], 'tg(a) = ' + str(np.tan(trace_param[0]))
        cosa = np.cos(trace_param[0])
        sina = np.sin(trace_param[0])
        return [(trace_param[1]) * x / (trace_param[2] * cosa - sina * x) for x in sample_positions]

    @staticmethod
    def zero_transformation():
        return [0, 1, 1]


class SignalExp(SignalModel):
    """Real, periodic bandlimited signal (exponential polynomial)"""

    def __init__(self, parameters, interval_length=2 * np.pi):
        super(SignalExp, self).__init__(parameters, interval_length)

    def value(self, x):
        v = 0
        for k in range(0, self.model_size):
            v += self.parameters[-(k + 1)] * np.cos(k * x)
        return v

    def path(self, start_pos, change, n=50):
        p = [start_pos]
        start_val = self.value(start_pos)
        new_parameters = np.copy(self.parameters)
        for i in range(1, n):
            s = SignalExp(new_parameters + (i * change) / (n - 1))
            r = optimize.newton(lambda x: s.value(x) - start_val, p[i - 1], fprime=s.derivative_value)
            assert (np.isclose(s.value(r), start_val))
            p.append(r)
        return p

    def get_samples(self, positions):
        samples = np.zeros(np.shape(positions))
        for k in range(0, self.model_size):
            samples += self.parameters[-(k + 1)] * np.cos(k * positions)
        return samples

    def derivative_value(self, x):
        v = 0
        for k in range(0, self.model_size):
            v -= k * self.parameters[-(k + 1)] * np.sin(k * x)
        return v

    def norm2(self, prm=None):
        if prm is None:
            prm = self.parameters
        return 2 * np.power(np.abs(prm), 2)


def next_zero(signal, x0, steps=1000, precision=10e-6, gamma=0.01):
    """
    Finds a position of zero of a signal using Newton's method

    Args:
        signal (SignalExp): bandlimited function which will be searched for a zero
        x0 (float): starting point for the search
        steps (int): maximal possible number of steps
        precision (float): maximal acceptable precession
        gamma (float): step size

    Returns:
        float: position of a zero

    """
    x = x0
    for i in range(0, steps):
        if abs(signal.value(x0)) < precision:
            break
        x -= gamma * signal.value(x) / signal.derivative_value(x)
    return x
