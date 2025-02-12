import numpy as np


def differentiate(u, dt):
    """
    Input:
    u exact function
    dt time step
    Return:
    d discrete derivative of the mesh function un
    
    Finds discrete derivative by centered differences for all the points in the    
    middle, and by forward/backward dfferences for the end points.
    """
    dt = float(dt)         #avoid integer division
    Nt = len(u) #nr of time intervals
    T = Nt*dt              #adjust T to fit time step dt
    d = np.zeros(Nt+1)     #array of u[n] values
    t = np.linspace(0,T,Nt+1) # time mesh
    d[0] = (u[1]-u[0])/dt  #starting point
    d[-1] = (u[-1]-u[-2])/dt #end point
    for i in range(1,Nt-1):  #the points in the middle
        d[i] = (u[i+1]-u[i-1])/(2*dt) 
    return d

def differentiate_vector(u, dt):
    """
    Input:
    u exact function
    dt time step
    Return:
    d discrete derivative of the mesh function un
    
    Finds discrete derivative by centered differences for all the points in the    
    middle, and by forward/backward dfferences for the end points.
    
    Using vectorization/array computing, for speeding up the calculations.
    """
    dt = float(dt)         #avoid integer division
    Nt = len(u)
    d = np.zeros(Nt+1)     #array of u[n] values
    d[0] = (u[1]-u[0])/dt  #starting point
    d[-1] = (u[-1]-u[-2])/dt #end point
    d[1:-2] = (u[2:]-u[0:-2])/(2*dt) #points in the middle
    return d
    
    

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    assert np.allclose(du1, du2)
    print("Works!")

if __name__ == '__main__':
    test_differentiate()
    