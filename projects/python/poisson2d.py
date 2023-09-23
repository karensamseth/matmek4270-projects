import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    """Solve Poisson's equation in 2D::

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
        self.dx = self.Lx / self.px.N
        self.dy = self.Ly / self.py.N
        x = self.x = np.linspace(0, self.px.L, self.px.N+1)
        y = self.y =np.linspace(0, self.py.L, self.py.N+1)
        return np.meshgrid(x, y, indexing='ij')

    def laplace(self):
        """Return a vectorized Laplace operator. Using Kronecker product to calculate."""
        D2x = (1./self.dx**2)*self.py.D2(self.Nx)
        D2y = (1./self.dy**2)*self.py.D2(self.Ny)
        return (sparse.kron(D2x, sparse.eye(self.Ny+1)) + sparse.kron(sparse.eye(self.Nx+1), D2y))

    def assemble(self, f=None):
        """Assemble coefficient matrix and right hand side vector

        Parameters
        ----------
        bc : 2-tuple of numbers
            The boundary conditions at x=0 and x=L
        f : Sympy Function
            The right hand side as a Sympy function

        Returns
        -------
        A : scipy sparse matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector
        """
        A = self.laplace()
        B = np.ones((self.px.N+1,self.py.N+1), dtype=bool) #Matrise for å finne indekser for boundary cond.
        B[1:-1,1:-1] = 0 #Matrise med 1 på kantene og 0 i midten.
        bnds = np.where(B.ravel() == 1)[0] #Finner hvilke indekser i B som er 1. RAVEL???
        xij, yij = self.create_mesh()
        b = sp.lambdify((x,y),f)(xij,yij)        
        return A.tocsr(), b

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

