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
        self.parameter_estimate = 0

    def solve(self):
        raise NotImplemented

    def test_error(self, signal):
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
        self.parameter_estimate = []*model_size

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
        self.parameter_estimate = np.zeros((self.model_size))

    def solve(self):

        for k in range(0, self.max_iterations):
            # if k%100==0 and k<5000: #brzydkie
            if k < 5:
                self.illustration.append(self.position_estimate)

            x = self.model_type.create_ls_matrix(self.position_estimate, self.model_size)
            self.parameter_estimate = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, self.samples))
            g = self.model_type.compute_ls_gradient(self.position_estimate, self.parameter_estimate, self.samples)
            if self.hold_edges:
                self.position_estimate[1:self.number_samples-1] -= self.beta * g[1:self.number_samples-1]
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

    def __init__(self, samples, model_size, model_type, start_pos,
                 show_plots=False, hold_edges=True, stopping_error=1.0e-6, beta=0.01, interval_length=1, max_iter=10000,
                fl=1.0, change_beta=False):
        super(ConstrainedALS,self).__init__(samples,model_size,model_type, show_plots,
                                            hold_edges, stopping_error, beta, interval_length)
        assert len(samples)==len(start_pos)
        self.position_estimate = start_pos
        self.start_positions = start_pos
        self.illustration_param = []
        self.tr_param = self.model_type.zero_transformation()
        if self.model_type == SecondSurfacePolynomial:
            self.tr_param[2]=fl
        self.max_iterations = max_iter

    def solve(self):
        sign = -1
        for k in range(0, self.max_iterations):
            # if k%100==0 and k<5000: #brzydkie
            # print "step %d" % k
            if k<5:
                self.illustration.append(self.position_estimate)
                self.illustration_param.append(self.tr_param)

            x = self.model_type.create_ls_matrix(self.start_positions, self.model_size, self.tr_param)
            self.parameter_estimate = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, self.samples))
            # is it random?
            self.error = np.linalg.norm(np.dot(x, self.parameter_estimate) - self.samples) / self.number_samples

            if self.error < self.stopping_error:
                print("error small enough after fitting parameters")
                break

            g = self.model_type.compute_ls_gradient(self.start_positions, self.parameter_estimate, self.samples,self.tr_param)
            # print "gradient: %f" % g
            # print "alpha: %f" % self.tr_param
            if np.max(np.abs(g*self.beta)) < np.finfo(float).eps:
                print("converged to local minimum after", k, "steps")
                break

            self.tr_param -= self.beta * g
            self.position_estimate = self.model_type.shifted_positions(self.start_positions, self.tr_param)
            error = np.linalg.norm(np.dot(x, self.parameter_estimate) - self.samples) / self.number_samples

            if error < self.stopping_error:
                print("error small enough after fitting positions")
                break

            if self.error < error:
                print("error:", self.error, "beta:", self.beta)
                if self.beta > 10*np.finfo(float).eps:
                    self.beta *=0.5

            if k>0 and sign != np.sign(g[0]):
                if self.show_plots:
                    print("jumped over minimum,", "beta:", self.beta)
                if self.beta > 10*np.finfo(float).eps:
                    self.beta *= 0.5

            sign = np.sign(g[0])


            if self.show_plots & (k % 10 == 0):
                print("first param change:", g[0], "beta:", self.beta)
                # print("alpha:", self.tr_param)
                pylab.stem(self.position_estimate, self.samples)
                time.sleep(0.01)
                pylab.pause(0.01)

            if k == self.max_iterations -1:
                print('force stop after', self.max_iterations, 'steps')



class InvertedLS(OrdinaryLS):
    """InvertedLS:
    for linear function, with assumption that samples are uniformly spaced with added gaussian error,
    we can exchange x and y and use OLS"""

    def __init__(self, samples, model_size, model_type, interval_length=1):
        super(InvertedLS, self).__init__(samples, model_size, model_type, interval_length)
        assert model_size == 2


    def solve(self):
        (self.position_estimate, self.samples) = (self.samples, self.position_estimate)
        (self.train_error, self.parameter_estimate) = self.solve()
        (self.position_estimate, self.samples) = (self.samples, self.position_estimate)
        pe = self.parameter_estimate
        self.parameter_estimate = [-pe[0]/pe[1], 1/pe[1]]
