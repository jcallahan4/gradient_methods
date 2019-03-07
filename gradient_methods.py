# solutions.py
"""Gradient Descent Methods
Jake Callahan

Iterative optimization methods choose a search direction and a step size at each
iteration. One choice for the search direction is the negative gradient, resulting
in the method of steepest descent. In practice this method is often slow to converge.
An alternative method, the conjugate gradient algorithm, uses a similar idea that
results in much faster convergence in some situations. In this program I implement
a method of steepest descent and two conjugate gradient methods, then apply them to
regression problems.
"""

import numpy as np
from scipy import optimize as opt
from scipy import linalg as la
from autograd import numpy as anp
from matplotlib import pyplot as plt

def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Initialize iteration number
    count = 0

    #Loop while x0 is bigger than tolerance to a certain number of iterations
    while la.norm(Df(x0), ord=np.inf) >= tol and count < maxiter:
        #Get optimal alpha for next step
        min_alpha = lambda a: f(x0 - a * Df(x0))
        alpha = opt.minimize_scalar(min_alpha)
        #Find next step x
        x0 = x0 - alpha.x * Df(x0)
        count += 1

    return x0, la.norm(Df(x0)) < tol, count

def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Initialize values
    n = len(x0)
    r0 = Q@x0 - b
    d0 = -r0
    r1 = x0
    count = 0

    #Loop while r1 is greater than tol and num of iterations is less that n
    while la.norm(r1) >= tol and count <= n:
        #Calculate next step values
        a0 = (r0@r0) / (d0@Q@d0)
        x1 = x0 + a0*d0
        r1 = r0 + a0*Q@d0
        beta1 = (r1@r1)/(r0@r0)
        d1 = -r1 + beta1*d0
        #Update values for next iteration
        x0, r0, d0 = x1, r1, d1
        count += 1

    return x1, la.norm(r1) < tol , count

def nonlinear_conjugate_gradient(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Initialize values
    r0 = -Df(x0)
    d0 = r0
    alpha0 = opt.minimize_scalar(lambda a: f(x0+a*d0)).x
    x1 = x0 + alpha0*d0
    r1 = x0
    count = 1
    #Loop while r1 is greater than tol and num of iterations is less that max iterations
    while la.norm(r1) >= tol and count <= maxiter:
        #Calculate next step values
        r1 = -Df(x1)
        beta1 = (r1@r1) / (r0@r0)
        d1 = r1 + beta1*d0
        alpha1 = opt.minimize_scalar(lambda a: f(x1 + a*d1)).x
        x2 = x1 + alpha1*d1
        #Update values for next iteration
        x0, r0, d0, alpha0 = x1, r1, d1, alpha1
        x1 = x2
        count += 1

    return x2, la.norm(r1) < tol, count-1

def linear_regression(filename="linregression.txt",
          x0=np.array([-3482258, 15, 0, -2, -1, 0, 1829])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    #Get data
    data = np.loadtxt(filename)
    #Initialize b and Q
    b = data[:,0]
    ones = np.ones(len(b))
    Q = np.column_stack((ones, data[:,1:]))
    #solve QTQx = b
    return conjugate_gradient(Q.T@Q,Q.T@b,x0)[0]

class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        #Create function for negative log likelihood
        L = lambda B: sum([anp.log(1+anp.exp(-(B[0]+B[1]*x[i]))) + (1-y[i])*(B[0]+B[1]*x[i])
                    for i in range(len(x))])
        #Minimize L
        beta = opt.fmin_cg(L,guess,disp=False)
        #Store values
        self.B0 = beta[0]
        self.B1 = beta[1]

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        #Calculate sigma
        return 1/(1+anp.exp(-(self.B0+self.B1*x)))

def challenger_damage(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    #Get data
    data = np.load(filename)
    x = data[:,0]
    y = data[:,1]
    #Get domain to fit on
    domain = np.linspace(30,100,71)
    #Fit and store prediction values using LogisticRegression class
    launch = LogisticRegression1D()
    launch.fit(x,y,guess)
    vals = [launch.predict(i) for i in domain]

    #Plot
    plt.plot(x, y, marker = 'o', linestyle = '', label = 'Previous damage')
    plt.plot(domain, vals, label = 'Probability')
    plt.plot(31, launch.predict(31), marker='o', label = 'P(Damage) of Launch')
    plt.ylabel('O-Ring Damage')
    plt.xlabel('Temperature')
    plt.title("Probability of O-Ring Damage")
    plt.legend()
    plt.show()
    return launch.predict(31)
