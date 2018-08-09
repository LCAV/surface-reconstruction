from plots import *
import time


class Solver(object):
    """General solver"""

    def __init__(self, samples, model_size, model_type, interval_length=1):
        self.samples = samples
        self.number_samples = len(samples)
        self.model_size = model_size
        self.model_type = model_type
        self.interval_length = interval_length
        self.position_estimate = np.linspace(0, self.interval_length, self.number_samples)
        self.parameter_estimate = np.zeros(1)

    def solve(self):
        """
        Finds parameters (different for different solvers),
        and stores results in it the fields of the solver.

        """
        raise NotImplemented

    def test_error(self, signal):
        """
        Args:
            signal (SignalModel): signal to compare with

        Returns:
            float: squared distance between stored signal and the given one
        """
        return signal.square_error(self.parameter_estimate)

    def get_position_estimates(self):
        return self.position_estimate

    def get_samples(self):
        return self.samples


class OrdinaryLS(Solver):
    """Ordinary Least Squares:
     assume that samples are uniquely spaced on [0,1]"""

    def __init__(self, samples, model_size, model_type, interval_length=1):
        super(OrdinaryLS, self).__init__(samples, model_size, model_type, interval_length)
        self.train_error = 0
        self.parameter_estimate = np.zeros(self.model_size)

    def solve(self):
        x = self.model_type.create_ls_matrix(self.position_estimate, self.model_size)
        estimated = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, self.samples))
        error = np.linalg.norm(np.dot(x, estimated) - self.samples) / self.number_samples
        (self.train_error, self.parameter_estimate) = (error, estimated)


class AlternatingLS(Solver):
    """Alternating least squares
    i. e. alternating least squares with gradient descent"""

    def __init__(self, samples, model_size, model_type,
                 show_plots=False, hold_edges=True, stopping_error=1.0e-6, beta=0.01, interval_length=1):
        super(AlternatingLS, self).__init__(samples, model_size, model_type, interval_length)
        self.beta = beta
        self.stopping_error = stopping_error
        self.show_plots = show_plots
        self.hold_edges = hold_edges
        self.max_iterations = 10000
        self.illustration = []
        self.train_error = 0.0
        self.parameter_estimate = np.zeros(self.model_size)

    def solve(self):

        for k in range(0, self.max_iterations):
            # if k%100==0 and k<5000: #brzydkie
            if k < 5:
                self.illustration.append(self.position_estimate)

            x = self.model_type.create_ls_matrix(self.position_estimate, self.model_size)
            self.parameter_estimate = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, self.samples))
            g = self.model_type.compute_ls_gradient(self.position_estimate, self.parameter_estimate, self.samples)
            if self.hold_edges:
                self.position_estimate[1:self.number_samples - 1] -= self.beta * g[1:self.number_samples - 1]
            else:
                self.position_estimate -= self.beta * g
            error = np.linalg.norm(np.dot(x, self.parameter_estimate) - self.samples) / self.number_samples

            if self.show_plots & (k % 10 == 0):
                print(error)
                pylab.stem(self.position_estimate, self.samples)
                time.sleep(0.001)
                pylab.pause(0.001)
            if error < self.stopping_error:
                print("error small enough")
                break


class ConstrainedALS(AlternatingLS):
    """Alternating least squares for constrained case
    i. e. alternating least squares with gradient descent,
    where parameters of matrix X depend on parameters (tr_param)"""

    def __init__(self, samples, model_size, model_type, start_pos,
                 show_plots=False, hold_edges=True, stopping_error=1.0e-9, beta=0.01, interval_length=1, max_iter=10000,
                 fl=1.0, angle=0, verbose=True, early_stopping=1.0e-16):
        super(ConstrainedALS, self).__init__(samples, model_size, model_type, show_plots,
                                             hold_edges, stopping_error, beta, interval_length)
        self.figure, self.axis = pylab.subplots(1, 3)
        assert len(samples) == len(start_pos)
        self.position_estimate = start_pos
        self.start_positions = start_pos
        self.illustration_param = []
        self.tr_param = self.model_type.zero_transformation()
        if self.model_type == SecondSurfacePolynomial:
            self.tr_param[2] = fl
            self.tr_param[0] = angle
        self.max_iterations = max_iter
        self.verb = verbose
        self.tr_params_over_time = []
        self.beta_over_time = []
        self.error_over_time = []
        self.error = np.infty
        self.early_stopping = early_stopping

    def solve(self):
        if self.show_plots:
            self.axis[0].set_title("beta")
            self.axis[1].set_title("error")
            self.axis[2].set_title("gradient")

        blocked = False
        for k in range(0, self.max_iterations):

            # solver is blocked if gradient step would take parameters outside safe intervals
            if not blocked:
                x = self.model_type.create_ls_matrix(self.start_positions, self.model_size, self.tr_param)
                try:
                    self.parameter_estimate = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, self.samples))
                except np.linalg.linalg.LinAlgError as lin_err:
                    print(lin_err)
                    print("angle:", self.tr_param[0])
                    break

                error = np.linalg.norm(np.dot(x, self.parameter_estimate) - self.samples) / self.number_samples

                if error < self.stopping_error:
                    if self.verb:
                        print("error small enough after fitting parameters")
                    break

                g = self.model_type.compute_ls_gradient(self.start_positions, self.parameter_estimate, self.samples,
                                                        self.tr_param)

                if np.max(np.abs(error - self.error)) < self.early_stopping:
                    if self.verb:
                        print("error stopped changing after", k, "steps")
                    break

            # instead of normalizing the gradient, reduce beta, to prevent parameters outside stable region
            # WARNING this is not a nice hack below:

            if self.tr_param[2] > np.abs(np.tan(self.tr_param[0] - self.beta * g[0])) and np.abs(
                    self.tr_param[0] - self.beta * g[0]) < (np.pi / 2.0):
                self.tr_param -= self.beta * g
                blocked = False
            else:
                self.beta *= 0.9
                blocked = True

            if self.beta < 1e-5:
                if self.verb:
                    print("beta became to small")
                break

            self.position_estimate = self.model_type.shifted_positions(self.start_positions, self.tr_param)

            error = np.linalg.norm(np.dot(x, self.parameter_estimate) - self.samples) / self.number_samples

            if error < self.stopping_error:
                if self.verb:
                    print("error small enough after fitting positions")
                break

            if self.error < error:
                if self.beta > 10 * np.finfo(float).eps:
                    self.beta *= 0.9

            self.error = error


            if self.show_plots:
                self.axis[0].plot(k, self.beta, 'go')
                self.axis[1].semilogy(k, self.error, 'ro')
                self.axis[2].semilogy(k, np.abs(g[0]), 'bo')
                pylab.pause(0.1)

            self.beta_over_time.append(self.beta)
            self.error_over_time.append(self.error)
            self.tr_params_over_time.append(self.tr_param[0])

            if k == self.max_iterations - 1:
                if self.verb:
                    print('force stop after', self.max_iterations, 'steps')
