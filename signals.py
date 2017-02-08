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
        raise NotImplementedError('get_samples must be implemented by the subclass')

    def square_error(self, signal2):
        raise NotImplementedError('square_error must be implemented by the subclass')

    def path(self, point, change, n=50):
        raise NotImplementedError('path must be implemented by the subclass')


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

    def square_error(self, signal2):
        diff_parameters = self.parameters - signal2.parameters
        return self.norm2(diff_parameters)

    def path(self, start_pos, change, n=50):
        start_pos = np.array(start_pos)
        p = [start_pos]
        value = self.get_samples(start_pos)
        new_pol = self.parameters
        new_pol[-1] -= value
        for i in range(1, n):
            new_pol += change
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
    def __init__(self, parameters, interval_length=1):
        super(FullSurfacePolynomial, self).__init__(parameters, interval_length)

    @classmethod
    def positions_derivative(cls, sample_positions, tr_parameter):
        coef = 1.0/(1.0 + tr_parameter[1])
        return np.array([[coef*x ** 2, coef*x]
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
    def __init__(self, parameters, interval_length=1):
        super(SecondSurfacePolynomial, self).__init__(parameters, interval_length)

    @classmethod
    def positions_derivative(cls, sample_positions, tr_parameter):
        cosa = np.cos(tr_parameter[0])
        sina = np.sin(tr_parameter[0])
        b = tr_parameter[1]
        f = tr_parameter[2]
        return np.array([[cls._denominator(s, f, cosa, sina) ** 2 * (f*sina + s*cosa ) *(b *s),
                          s*cls._denominator(s,f,cosa, sina), 0]
                         for s in sample_positions])

    @staticmethod
    def _denominator(s, f, cosa, sina):
        return 1.0/(f*cosa - s * sina)

    @staticmethod
    def shifted_positions(sample_positions, trace_param):
        assert trace_param[1]>0, 'b = ' + str(trace_param[1])
        assert abs(trace_param[0]) < (np.pi/2.0), 'a = ' + str(trace_param[0])
        assert np.tan(trace_param[0]) < trace_param[2], 'a = ' + str(trace_param[0])
        cosa = np.cos(trace_param[0])
        sina = np.sin(trace_param[0])
        return [(trace_param[1]) * x / (trace_param[2]*cosa - sina * x) for x in sample_positions]

    @staticmethod
    def zero_transformation():
        return [0, 1, 1]


class SignalExp(SignalModel):
    def __init__(self, parameters, interval_length=2 * np.pi):
        super(SignalExp, self).__init__(parameters, interval_length)

    def value(self, x):
        v = 0
        for k in range(0, self.model_size):
            v += self.parameters[k] * np.cos(k * x)
        return v

    def path(self, start_pos, change, n=50):
        p = [start_pos]
        new_parameters = self.parameters
        new_parameters[0] -= self.value(start_pos)
        for i in range(1, n):
            s = SignalExp(new_parameters + i * change)
            r = optimize.newton(s.value, p[i - 1], fprime=s.derivative_value)
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
            v -= k * self.parameters[k] * np.sin(k * x)
        return v


def next_zero(signal, x0, steps=1000, precision=10e-6, gamma=0.01):
    x = x0
    for i in range(0, steps):
        if abs(signal.value(x0)) < precision:
            break
        x -= gamma * signal.value(x) / signal.derivative_value(x)
    return x