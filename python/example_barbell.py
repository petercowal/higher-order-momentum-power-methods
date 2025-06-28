import numpy as np
from powermethods import powermethod, momentum, momentum2, momentum_dynamic, parameter_search
import matplotlib.pyplot as plt

# generate barbell graph transition matrix
N = 160
p = 1/10

X1 = np.ones((N, N)) * (np.random.rand(N, N) < p)
X2 = np.ones((N, N)) * (np.random.rand(N, N) < p)

B = np.block([
    [X1, np.zeros((N, N))],
    [np.zeros((N, N)), X2]
])

B[N,N-1] = 1
B[N-1,N] = 1

Dinv = np.diag(1/np.sum(B, axis=0))

A = B @ Dinv

plt.figure()
plt.imshow(A)
plt.colorbar()

# iteration count
iter = 500

# compute eigenvalues and plot them for reference
eigs, U = np.linalg.eig(A)
# sort them from largest to smallest magnitude
idx = np.argsort(-np.abs(eigs))
eigs = eigs[idx]
U = U[:, idx]

U1 = U[:, 0]

plt.subplots()

theta = np.linspace(0, 2*np.pi, 60)
curve = (2/3*np.exp(theta*1j) + 1/3*np.exp(-theta*2j))

plt.plot(np.cos(theta), np.sin(theta), ':', label = "$|z| = 1$")
plt.plot(np.real(curve), np.imag(curve), '--', label = r"$\gamma^{(3)}$")

plt.plot(np.real(eigs), np.imag(eigs), '.', label = r"$\lambda_n$")
plt.legend()
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.axis('square')

xinit = (np.random.rand(2*N)+1j*np.random.rand(2*N)).reshape(-1,1)

# momentum parameters
beta = parameter_search(A, xinit, iter, np.linspace(0.2, 0.25, 20), U1)
beta2 = 4*np.abs(eigs[1])**3/27

# relative gap between magnitude of 1st and 2nd eigenvalues
# used to determine theoretical rate of convergence
spectral_gap = np.abs(eigs[0])/np.abs(eigs[1]) - 1
print(f"spectral gap = {spectral_gap}")



# apply different power methods for comparison purposes
x1, errs1 = powermethod(A, xinit, iter, xtrue=U1)
print("power method err=\n",errs1[-1])

x2, errs2 = momentum(A, xinit, iter, beta, xtrue=U1)
print("momentum method err=\n",errs2[-1])

x3, errs3 = momentum2(A, xinit, iter, beta2, xtrue=U1)
print("order 2 momentum method err=\n",errs3[-1])

x4, errs4 = momentum_dynamic(A, xinit, iter, xtrue=U1)
print("order 2 dynamic momentum method err=\n",errs4[-1])

# plot results
iters = np.arange(iter+1)
plt.subplots()
plt.semilogy(iters, errs1, '-', marker='x', markevery=iter//10, label = 'power method')
plt.semilogy(iters, errs2, '-', marker='s', markevery=iter//10, label = f'momentum ($\\beta = {beta:.3f}$)')
plt.semilogy(iters, errs3, '-', marker='o', markevery=iter//10, label = f'order 2 momentum ($\\beta = {beta2:.3f}$)')
plt.semilogy(iters, errs4, '-', marker='*', markevery=iter//10, label = 'order 2 dyn momentum')

# plot theoretical asymptotic convergence as well
asympt = np.exp(-iters*np.sqrt(spectral_gap))
plt.semilogy(iters, asympt * errs3[-1]/asympt[-1], '--', label = r"$O(e^{-n\sqrt{\varepsilon}})$")
plt.ylim(1e-10, 10)

plt.legend()
plt.xlabel("n")
plt.ylabel("relative error")
plt.tight_layout()
#plt.savefig("relerr_barbell.eps")
plt.show()
