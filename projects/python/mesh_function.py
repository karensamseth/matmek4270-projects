import numpy as np

def mesh_function(f, t):
    """
    Input: 
    f Python function
    t array of mesh points
    Nt number of mesh points in t
    Return:
    an array with mesh point values for f.
    """
    dt = 0.1
    Nt = int(t[-1]/dt)
    T = Nt*dt
    u = np.zeros(Nt+1)
    t_i = t[0]
    for i in range(Nt+1):
        u[i] = f[i]         #hva er poenget med denne?
        t_i = t_i + dt
    return u

def func(t):                #burde jeg heller bare sette denne rett inn i mesh_function? Denne diskretiserer jo også...
    """
    Input:
    t array of mesh points
    Returns:
    f an array with points, calculated by
        f(t) = e**(-t) for 0<=t<=3
        f(t) = e**(-3t) for 3<t<=4
    """
    f = np.zeros(len(t))
    for i in range(len(t)):
        if t[i]<=0 and t[i]>=3:
            f[i] = np.exp(-t[i])
        elif t[i]<3 and t[i]>=4:
            f[i] = np.exp(-3*t[i])
    return f

t = np.linspace(0,4)
print(mesh_function(func(t),t))

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

