import numpy as np


def mesh_function(f, t):
    t0 = t(0)
    u0 = f(0)
    I = u0
    T = t(-1)
    Nt = len(t)
    dt = T/Nt
    u = np.zeros(Nt+1)
    u = f(t)
    return u

def func(t):
    for i in range(len(t)+1):
        if t[i]<=0 & t[i]>=3:
            f_mesh[i] = e**(-t[i])
        elif t[i]<=3 & t[i]>=4:
            f_mesh[i] = e**(-3*t[i])
    return f_mesh

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
