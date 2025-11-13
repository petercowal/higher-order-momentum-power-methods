import numpy as np


def measure_error(x1, xtrue):
    x1 = x1.flatten()
    xtrue = xtrue.flatten()
    return np.linalg.norm(np.vdot(x1,xtrue)/np.vdot(x1,x1)*x1-xtrue)/np.linalg.norm(xtrue)

# Algorithm 1 Power Method from https://arxiv.org/abs/2507.01885
def powermethod(A, v0, n, xtrue):
    errs = np.zeros(n+1)
    errs[0] = measure_error(v0, xtrue)
    for i in range(n):
        h0 = np.linalg.norm(v0)
        x0 = v0/h0
        v1 = A @ x0
        v0 = v1
        errs[i+1] = measure_error(v1, xtrue)
    return v1/np.linalg.norm(v1), errs

# Algorithm 1.2. Power iteration with momentum from https://arxiv.org/abs/2403.09618
def momentum(A, v0, n, beta, xtrue):
    errs = np.zeros(n+1)
    errs[0] = measure_error(v0, xtrue)

    h0 = np.linalg.norm(v0)
    x0 = v0/h0
    v1 = A @ x0
    errs[1] = measure_error(v1, xtrue)

    h1 = np.linalg.norm(v1)
    x1 = v1/h1
    v2 = A @ x1
    errs[2] = measure_error(v2, xtrue)

    for i in range(3,n+1):
        u2 = v2 - (beta/h1)*x0
        h2 = np.linalg.norm(u2)
        x2 = u2/h2
        v3 = A @ x2

        v2 = v3
        x0 = x1
        x1 = x2
        h0 = h1
        h1 = h2

        errs[i] = measure_error(v2, xtrue)
    return v2/np.linalg.norm(v2), errs

# Algorithm 2 Deltoid Momentum Power Method from https://arxiv.org/abs/2507.01885
def momentum2(A, v0, n, beta, xtrue):
    errs = np.zeros(n+1)
    errs[0] = measure_error(v0, xtrue)

    h0 = np.linalg.norm(v0)
    x0 = v0/h0

    v1 = 2/3*A @ x0
    h1 = np.linalg.norm(v1)
    x1 = v1/h1
    errs[1] = measure_error(x1, xtrue)

    v2 = 2/3*A @ x1
    h2 = np.linalg.norm(v2)
    x2 = v2/h2
    errs[2] = measure_error(x2, xtrue)

    for i in range(3,n+1):
        v3 = A @ x2
        u3 = v3 - beta/(h2*h1)*x0
        h3 = np.linalg.norm(u3)
        x3 = u3/h3

        x0 = x1
        x1 = x2
        x2 = x3
        h1 = h2
        h2 = h3

        errs[i] = measure_error(x3, xtrue)
    return x3, errs


# Algorithm 3 Dynamic Deltoid Momentum Power Method from https://arxiv.org/abs/2507.01885
def momentum_dynamic(A, v0, n, xtrue):
    errs = np.zeros(n+1)
    errs[0] = measure_error(v0, xtrue)

    h0 = np.linalg.norm(v0)
    x0 = v0/h0


    v1 = 2/3*A @ x0
    h1 = np.linalg.norm(v1)
    x1 = v1/h1
    errs[1] = measure_error(x1, xtrue)

    v2 = 2/3*A @ x1
    h2 = np.linalg.norm(v2)
    x2 = v2/h2
    errs[2] = measure_error(x2, xtrue)

    nu1 = np.vdot(v2,x1)
    d1 = np.linalg.norm(v2 - nu1*x1)

    for i in range(3,n+1):

        v3 = A @ x2
        nu2 = np.vdot(v3,x2)
        d2 = np.linalg.norm(v3 - nu2*x2)
        rho1 = np.min((d2/d1, 1))
        r2 = 1/(np.log(rho1)**2 + 1)

        beta = 4*(nu2*r2)**3/27
        u3 = v3 - beta/(h2*h1)*x0
        h3 = np.linalg.norm(u3)
        x3 = u3/h3

        x0 = x1
        x1 = x2
        x2 = x3
        h1 = h2
        h2 = h3
        d1 = d2

        errs[i] = measure_error(x3, xtrue)
    return x3, errs


