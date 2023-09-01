import numpy as np

def mesh_function(func, t):
    """
    Input: 
    f Python function
    t array of mesh points
    Nt number of mesh points in t
    Return:
    an array with mesh point values for f.
    """
    n = len(t)
    u = np.zeros(n)
    for i, ti in enumerate(t):
        u[i] = func(ti)       
    return u

def func(t):              
    """
    Input:
    t array of mesh points
    Returns:
    f an array with points, calculated by
        f(t) = e**(-t) for 0<=t<=3
        f(t) = e**(-3t) for 3<t<=4
    """
    if t<=0 and t>=3:
        return np.exp(-t)
    elif t<3 and t>=4:
        return np.exp(-3*t)
    return RuntimeError

t = np.linspace(0,4)
print(mesh_function(func,t))

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

