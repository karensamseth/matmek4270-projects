import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)

    def create_mesh(self):
        """Uniform discretization of the line [0, L]

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in x-direction
        Ny : int
            The number of uniform intervals in y-direction

        Returns
        -------
        x : array
            The mesh
        y : array
            The mesh
        """
        self.Nx = Nx
        self.Ny = Ny
        self.dx = self.Lx / Nx
        self.dy = self.Ly / Ny
        self.x = np.linspace(0, self.Lx, self.Nx+1)
        self.y = np.linspace(0, self.Ly, self.Ny+1)
        mesh = np.meshgrid(self.x, self.y, indexing='ij')
        return self.x, self.y, mesh

    def laplace(self):
        """Return a vectorized Laplace operator"""
        D2x = (1./self.dx**2)*self.D2(self.Nx)
        D2y = (1./self.dy**2)*self.D2(self.Ny)
        return (sparse.kron(D2x, sparse.eye(self.Ny+1)) + sparse.kron(sparse.eye(self.Nx+1), D2y))

    def assemble(self, f=None):
        """Return assemble coefficient matrix A and right hand side vector b"""
        raise NotImplementedError

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        raise NotImplementedError

    def __call__(self, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(f=f)
        return sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))

def test_poisson2d():
    assert False

