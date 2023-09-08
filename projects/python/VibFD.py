"""
In this module we study the vibration equation

    u'' + w^2 u = f, t in [0, T]

where w is a constant and f(t) is a source term assumed to be 0.
We use various boundary conditions.

FUNKER IKKE ENDA!

"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy import sparse

t = sp.Symbol('t')

class VibSolver: #fra Michael
    """
    Solve vibration equation::

        u'' + w**2 u = f,

    """
    def __init__(self, Nt, T, w=0.35, I=1):
        """
        Parameters
        ----------
        Nt : int
            Number of time steps
        T : float
            End time
        I, w : float, optional
            Model parameters
        """
        self.I = I
        self.w = w
        self.T = T
        self.set_mesh(Nt)

    def set_mesh(self, Nt):
        """Create mesh of chose size

        Parameters
        ----------
        Nt : int
            Number of time steps
        """
        self.Nt = Nt
        self.dt = self.T/Nt
        self.t = np.linspace(0, self.T, Nt+1)

    def ue(self):
        """Return exact solution as sympy function
        """
        return self.I*sp.cos(self.w*t)

    def u_exact(self):
        """Exact solution of the vibration equation

        Returns
        -------
        ue : array_like
            The solution at times n*dt
        """
        return sp.lambdify(t, self.ue())(self.t)

    def l2_error(self):
        """Compute the l2 error norm of solver

        Returns
        -------
        float
            The l2 error norm
        """
        u = self()
        ue = self.u_exact()
        return np.sqrt(self.dt*np.sum((ue-u)**2))

    def convergence_rates(self, m=4, N0=32):
        """
        Compute convergence rate

        Parameters
        ----------
        m : int
            The number of mesh sizes used
        N0 : int
            Initial mesh size

        Returns
        -------
        r : array_like
            The m-1 computed orders
        E : array_like
            The m computed errors
        dt : array_like
            The m time step sizes
        """
        E = []
        dt = []
        self.set_mesh(N0) # Set initial size of mesh
        for m in range(m):
            self.set_mesh(self.Nt+10)
            E.append(self.l2_error())
            dt.append(self.dt)
        r = [np.log(E[i-1]/E[i])/np.log(dt[i-1]/dt[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(dt)

    def test_order(self, m=5, N0=100, tol=0.1):
        r, E, dt = self.convergence_rates(m, N0)
        assert np.allclose(np.array(r), self.order, atol=tol)

class VibHPL(VibSolver):
    """
    Second order accurate recursive solver

    Boundary conditions u(0)=I and u'(0)=0
    """
    order = 2

    def __call__(self): #fra Michael
        u = np.zeros(self.Nt+1)
        u[0] = self.I #u(0)=I
        u[1] = u[0] - 0.5*self.dt**2*self.w**2*u[0] #u'(0)=0
        for n in range(1, self.Nt):
            u[n+1] = 2*u[n] - u[n-1] - self.dt**2*self.w**2*u[n]
        return u

class VibFD2(VibSolver):
    """
    Second order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 2

    def __init__(self, Nt, T, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi
        assert T.is_integer() and T % 2 == 0

    def __call__(self):
        D2 = sparse.diags([np.ones(self.Nt), np.full(self.Nt+1, -2), np.ones(self.Nt)], np.array([-1, 0, 1]), (self.Nt+1, self.Nt+1), 'lil')
        D2 *= (1/self.dt**2) 
        Id = sparse.eye(self.Nt+1)
        A = (D2 + self.w**2*Id).tolil()
        A[0,:3] = 1, 0, 0 #u(0)=I
        A[self.Nt,-3:] = 0, 0, 1 #u(T)=I
        b = np.zeros(self.Nt+1)
        b[0] = self.I
        b[-1] = self.I
        u = sparse.linalg.spsolve(A.tocsr(), b)
        return u

class VibFD3(VibSolver):
    """
    Second order accurate solver using mixed Dirichlet and Neumann boundary
    conditions::

        u(0)=I and u'(T)=0

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 2

    def __init__(self, Nt, T, w=0.35, I=1):
        VibSolver.__init__(self, Nt, T, w, I)
        T = T * w / np.pi
        assert T.is_integer() and T % 2 == 0

    def __call__(self):
        D2 = sparse.diags([np.ones(self.Nt), np.full(self.Nt+1, -2), np.ones(self.Nt)], np.array([-1, 0, 1]), (self.Nt+1, self.Nt+1), 'lil')
        D2 *= (1/self.dt**2) 
        Id = sparse.eye(self.Nt+1)
        A = D2 + self.w**2*Id
        A[0,:3] = 1, 0, 0 #u(0)=b(0)
        A[self.Nt,-3:] = 0, 0, 1 #u(T)=b(T)
        b = np.zeros(self.Nt+1)
        b[0] = self.I #u(0)=I
        b[-1] = (2*u[-2])/(2-self.dt**2*self.w**2) #u'(T)=0
        A.toarray()
        u = sparse.linalg.spsolve(A, b)
        return u

class VibFD4(VibFD2):
    """
    Fourth order accurate solver using boundary conditions::

        u(0)=I and u(T)=I

    The boundary conditions require that T = n*pi/w, where n is an even integer.
    """
    order = 4

def __call__(self):
    D4 = sparse.diags([-np.ones(self.Nt), 16.*np.ones(self.Nt), np.full(self.Nt+1, -30), 16.*np.ones(self.Nt), -np.ones(self.Nt)], np.array([-2,-1, 0, 1,2]), (self.Nt+1, self.Nt+1), 'lil')
    D4 *= (1/self.dt**2) 
    Id = sparse.eye(self.Nt+1)
    A = D4 + self.w**2*Id
    A[0,:6] = 1, 0, 0, 0, 0, 0 #u(0)=b(0)
    A[1,:6] = 10, -15, -4, 14, -6, 1 #skewed scheme for n=1
    A[self.Nt-1,-6:] = 1, -6, 14, -4, -15, 10 #skewed scheme for n=N-1
    A[self.Nt,-3:] = 0, 0, 1 #u(T)=b(T)
    b = np.zeros(self.Nt+1)
    b[0] = self.I #u(0)=I
    b[-1] = self.I #u(T)=I
    A.toarray()
    u = sparse.linalg.spsolve(A, b)
    return u

def test_order(): #fra Michael
    w = 0.35
    VibHPL(8, 2*np.pi/w, w).test_order()
    VibFD2(8, 2*np.pi/w, w).test_order()
    VibFD3(8, 2*np.pi/w, w).test_order()
    VibFD4(8, 2*np.pi/w, w).test_order(N0=20)

if __name__ == '__main__':
    #test_order()
    w = 0.35
    v = VibFD3(32, 2*np.pi/w, w)
    print(v.l2_error())
