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
    n = len(t)
    u = np.zeros(n)
    for i, ti in enumerate(t):
        u[i] = f(ti)  #finner funksjonsverdien til hvert tidspunkt, setter i array u
    return u

def func(t):              
    """
    Input:
    t ett tidspunkt
    Returns:
    f en funksjonsverdi,
        f(t) = exp(-t) for 0<=t<=3
        f(t) = exp(-3t) for 3<t<=4
    """
    if t>=0 and t<=3:
        return np.exp(-t)
    elif t>3 and t<=4:
        return np.exp(-3*t)
    raise RuntimeError

#t = np.linspace(0,4)
#print(mesh_function(func,t))

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
    print("Works!")

if __name__ == "__main__":    #kjører testfunksjonen
    test_mesh_function()
