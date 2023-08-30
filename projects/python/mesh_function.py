import numpy as np


def mesh_function(f, t):
    dt = 0.1
    Nt = int(t(-1)/dt)
    T = Nt*dt
    u = np.zeros(Nt+1)
    t_i = t(0)
    for i in range(Nt+1):
        u[i] = f(t_i)
        t_i = t_i + dt
    return u

def func(t):
    f_mesh = np.zeros(len(t))
    for i in range(len(t)+1):
        if t[i]<=0 & t[i]>=3:
            f_mesh[i] = e**(-t[i])
        elif t[i]<=3 & t[i]>=4:
            f_mesh[i] = e**(-3*t[i])
    return f_mesh

t = linspace(0,4)
print(mesh_function(func(t),t))

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
