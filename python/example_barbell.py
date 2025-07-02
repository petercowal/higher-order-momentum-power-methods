import numpy as np
from powermethods import powermethod, momentum, momentum2, momentum_dynamic
import matplotlib.pyplot as plt
import scipy.sparse as spspr
np.random.seed(0)

# generate barbell graph transition matrix
N = 16000
p = 1/1000
print("generating B",end='')
rows = [N,N-1]
cols = [N-1,N]
data = [1,1]
for i in range(N):
    randos = np.random.rand(N)
    if i%1000 == 0:
        print('.',end='',flush=True)
    for j in range(N):
        if randos[j] < p:
            rows.append(i)
            cols.append(j)
            data.append(1)
for i in range(N,2*N):
    randos = np.random.rand(N)
    if i%1000 == 0:
        print('.',end='',flush=True)
    for j in range(N,2*N):
        if randos[j-N] < p:
            rows.append(i)
            cols.append(j)
            data.append(1)
B = spspr.coo_array((data, (rows, cols)), shape=(2*N, 2*N)).tocsr()

print("done!")
print("normalizing columns")
Dinv = spspr.diags(1/np.sum(B, axis=0))

A = B @ Dinv

# iteration count
iter = 5000

print("finding eigenvalues")
# compute eigenvalues and plot them for reference
eigs, U = spspr.linalg.eigs(A)
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
beta = np.abs(eigs[1])**2/4
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
asympt = np.pow(1+np.sqrt(spectral_gap), -iters)
plt.semilogy(iters, asympt * errs3[-1]/asympt[-1], '--', label = r"$O((1+\sqrt{\varepsilon})^{-N})$")
plt.ylim(1e-8, 10)

plt.legend()
plt.xlabel("n")
plt.ylabel("relative error")
plt.tight_layout()
plt.savefig("relerr_barbell.eps")
plt.show()
