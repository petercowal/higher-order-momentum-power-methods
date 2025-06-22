import numpy as np


def measure_error(x1, xtrue):
    x1 = x1.flatten() 
    xtrue = xtrue.flatten()
    return np.linalg.norm(np.vdot(x1,xtrue)/np.vdot(x1,x1)*x1-xtrue)/np.linalg.norm(xtrue)

def powermethod(A, xinit, n, xtrue = None):
    errs = np.zeros(n+1)
    x0 = xinit
    errs[0] = measure_error(x0, xtrue)
    for i in range(n):
        x1 = A @ x0
        x0 = x1
        errs[i+1] = measure_error(x1, xtrue)
    x1 = x1/np.linalg.norm(x1)
    return x1, errs

# implementation of a static momentum power method
def momentum(A, xinit, n, beta = 0.25, xtrue = None):
    errs = np.zeros(n+1)
    x0 = xinit
    x1 = 0.5*(A @ x0)
    errs[0] = measure_error(x0, xtrue)
    errs[1] = measure_error(x1, xtrue)
    for i in range(n-1):
        x2 = A @ x1 - beta*x0
        x0 = x1
        x1 = x2
        errs[i+2] = measure_error(x2, xtrue)
    x2 = x2/np.linalg.norm(x2)
    return x2, errs

# implementation of a static order 2 momentum power method
def momentum2(A, xinit, n, gamma = 4/27, xtrue = None):
    errs = np.zeros(n+1)
    x0 = xinit
    x1 = 2/3*(A @ x0)
    x2 = 2/3*(A @ x1)
    errs[0] = measure_error(x0, xtrue)
    errs[1] = measure_error(x1, xtrue)
    errs[2] = measure_error(x2, xtrue)
    for i in range(n-2):
        x3 = A @ x2 - gamma * x0
        x0 = x1
        x1 = x2
        x2 = x3
        errs[i+3] = measure_error(x3, xtrue)
    x3 = x3/np.linalg.norm(x3)
    return x3, errs

# implementation of a dynamic order 2 momentum power method
def momentum_dynamic(A, xinit, n, xtrue = None):
    errs = np.zeros(n+1)
    v0 = xinit
    v1 = (A @ v0)
    h1 = np.linalg.norm(v1)
    v2 = (A @ v1)
    h2 = np.linalg.norm(v2)

    d1 = np.linalg.norm(v1)
    d2 = np.linalg.norm(v2)
    r = np.min((d2/d1, 1))
    x0 = v0/np.linalg.norm(v0)
    x1 = v1/h1
    x2 = v2/h2
    nu = np.sum(v2*x2)

    errs[0] = measure_error(v0, xtrue)
    errs[1] = measure_error(v1, xtrue)
    errs[2] = measure_error(v2, xtrue)

    for i in range(n-2):
        beta = (nu*r)**3*4/27
        #u = A @ v2 - (beta/(h1*h2))*x0
        u = v2 - (beta/(h1*h2))*x0
        h1 = h2
        h2 = np.linalg.norm(u)
        x3 = u/h2
        v3 = A @ x3
        nu = np.sum(np.conj(v3)*x3)
        d1 = d2
        d2 = np.linalg.norm(v3 - nu*x3)
        rho = np.min((d2/d1, 1))
        r = 1/(np.log(rho)**2 + 1)
        v0 = v1
        v1 = v2
        v2 = v3
        x0 = x1
        x1 = x2
        x2 = x3

        errs[i+3] = measure_error(v3, xtrue)
    v3 = v3/np.linalg.norm(v3)
    return v3, errs
