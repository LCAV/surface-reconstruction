import numpy.random as nr
from signals import *

# TODO get read of "get thing" or add more "get thing"
# TODO unify LARGE names and normal names
class DeterministicSampler(object):
    """general sampler:
     keeps sample positions an compute sample values"""

    def __init__(self, signal, number_samples, interval_length=1):
        self.number_samples = number_samples
        self.interval_length = interval_length
        self.SAMPLE_POSITIONS = self._make_positions()
        self.SAMPLES = self._make_samples(signal)

    def _make_positions(self):
        return np.linspace(0, self.interval_length, self.number_samples)

    def _make_samples(self, signal):
        return signal.get_samples(self.get_positions())

    def get_positions(self):
        return self.SAMPLE_POSITIONS

    def get_samples(self):
        return self.SAMPLES


# TODO: use some seed
class GaussianSampler(DeterministicSampler):
    def __init__(self, signal, number_samples, interval_length=1, hold_edges=True, sigma=None):
        if sigma is None:
            sigma = (1.0*interval_length)/number_samples
        self.SIGMA = sigma
        self.hold_edges = hold_edges
        super(GaussianSampler, self).__init__(signal, number_samples)
        self.SAMPLES = self._make_samples(signal)

    def _make_positions(self):
        positions = np.linspace(0, self.interval_length, self.number_samples)
        positions += self.SIGMA * nr.randn(self.number_samples, 1).reshape(self.number_samples)
        positions = np.sort(positions)
        if self.hold_edges:
            positions[0] = 0
            positions[self.number_samples-1] = self.interval_length
        return positions


class SurfaceSampler(GaussianSampler):
    def __init__(self, signal, number_samples, surf_params, interval_length=1, sigma=None):
        self.surf_params = surf_params
        super(SurfaceSampler, self).__init__(signal, number_samples, interval_length, False, sigma)
        self.SAMPLE_POSITIONS = self._make_positions()
        self.SAMPLES = self._make_samples(signal)

    def _make_samples(self, signal):
        true_pos = signal.shifted_positions(self.SAMPLE_POSITIONS, self.surf_params)
        return  signal.get_samples(true_pos)


class GaussianSamplerFree(GaussianSampler):
    def __init__(self, signal, number_samples, interval_length=1, *sigma):
        if sigma == ():
            sigma = (1.0*interval_length)/number_samples
        super(GaussianSamplerFree, self).__init__(signal, number_samples, interval_length, False, sigma)
