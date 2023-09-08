"""Matrise-testing, ikke kjør"""

#VibFD2
#som matrise:
def __call__(self):
    D2 = sparse.diags([np.ones(N), np.full(N+1, -2), np.ones(N)], np.array([-1, 0, 1]), (N+1, N+1), 'lil')
    D2 *= (1/dt**2) 
    Id = sparse.eye(self.Nt+1)
    A = D1 + self.w**2*Id
    A[0,:3] = 1, 0, 0 #u(0)=I
    A[self.Nt+1,-3:] = 0, 0, 1 #u(T)=I
    b = np.zeros(self.Nt+1)
    b[0] = self.I
    b[-1] = self.I
    A.toarray()
    u = sparse.linalg.spsolve(A, b)
    return u

#som for-løkke:
def __call__(self):
    u = np.zeros(self.Nt+1)
    u[0] = self.I #spesifikk grense, u(0)=I
    for n in range(1, self.Nt): #samme som i VibHPL
        u[n+1] = 2*u[n] - u[n-1] - self.dt**2*self.w**2*u[n]
    u[-1] = self.I #spesifikk grense, u(T)=I
    return u


#VibFD3:
#som matrise:
def __call__(self):
    D2 = sparse.diags([np.ones(N), np.full(N+1, -2), np.ones(N)], np.array([-1, 0, 1]), (N+1, N+1), 'lil')
    D2 *= (1/dt**2) 
    Id = sparse.eye(self.Nt+1)
    A = D1 + self.w**2*Id
    A[0,:3] = 1, 0, 0 #u(0)=b(0)
    A[self.Nt+1,-3:] = 0, 0, 1 #u(T)=b(T)
    b = np.zeros(self.Nt+1)
    b[0] = self.I #u(0)=I
    b[-1] = (2*u[-2])/(2-self.dt**2*self.w**2) #u'(T)=0
    A.toarray()
    u = sparse.linalg.spsolve(A, b)
    return u

#som for-løkke:
def __call__(self):
    u = np.zeros(self.Nt+1)
    u[0] = self.I #u(0)=0
    for n in range(1, self.Nt): #samme som i VibHPL
        u[n+1] = 2*u[n] - u[n-1] - self.dt**2*self.w**2*u[n]
    u[-1] = (2*u[-2])/(2-self.dt**2*self.w**2) #u'(T)=0
    return u



#VibFD4:
#som matrise:
def __call__(self):
    D2 = sparse.diags([-np.ones(N), 16.*np.ones(N), np.full(N+1, -30), 16.*np.ones(N), -np.ones(N)], np.array([-2,-1, 0, 1,2]), (N+1, N+1), 'lil')
    D2 *= (1/dt**2) 
    Id = sparse.eye(self.Nt+1)
    A = D1 + self.w**2*Id
    A[0,:6] = 1, 0, 0, 0, 0, 0 #u(0)=b(0)
    A[1,:6] = 10, -15, -4, 14, -6, 1 #skewed scheme for n=1
    A[self.Nt,-6:] = 1, -6, 14, -4, -15, 10 #skewed scheme for n=N-1
    A[self.Nt+1,-3:] = 0, 0, 1 #u(T)=b(T)
    b = np.zeros(self.Nt+1)
    b[0] = self.I #u(0)=I
    b[-1] = self.I #u(T)=I
    A.toarray()
    u = sparse.linalg.spsolve(A, b)
    return u

#som for-løkke
def __call__(self):
    u = np.zeros(self.Nt+1)
    u[0] = self.I #spesifikk grense, u(0)=I
    l = 15-12*self.dt**2*self.w**2 #konstant til n=1 og n=N-1
    u[1] = (u[5]-6*u[4]+14*u[3]-4*u[2]+10*u[0])/l #skewed scheme
    k = -30+12*self.det**2*self.w**2 #konstant til de mellom
    for n in range(2, self.Nt-1): 
        u[n+2] = 16*u[n+1]+k*u[n]+15*u[n-1]-u[n-2] #4.ordens sentral d.s.
    u[-1] = self.I #spesifikk grense, u(T)=I
    u[-2] = (10*u[-1]-4*u[-3]+14*u[-4]-6*u[-5]+u[-6])/l #skewed scheme
    return u