import numpy as np
from powermethods import powermethod, momentum, momentum2, momentum_dynamic
import matplotlib.pyplot as plt

# simple 4x4 example with eigenvalues that lie within a deltoid
N = 4
A = np.array([[1.01, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1/3], [0, 0, 1/3, 0]])
U1 = np.zeros(N)
U1[0] = 1

# iteration count
iter = 200

# relative gap between magnitude of 1st and 2nd eigenvalues
# used to determine theoretical rate of convergence
spectral_gap = 0.01

print("A=\n",A)

xinit = (np.random.rand(N)+1j*np.random.rand(N)).reshape(-1,1)
print("xinit=\n",xinit)

# momentum parameters
beta = 1/4
beta2 = 4/27

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
plt.semilogy(iters, errs3, '-', marker='o', markevery=iter//10, label = 'order 2 momentum ($\\beta = 4/27$)')
plt.semilogy(iters, errs4, '-', marker='*', markevery=iter//10, label = 'order 2 dyn momentum')

# plot theoretical asymptotic convergence as well
asympt = np.pow(1+np.sqrt(spectral_gap), -iters)
plt.semilogy(iters, asympt * errs3[-1]/asympt[-1], '--', label = r"$O((1+\sqrt{\varepsilon})^{-N})$")
plt.ylim(1e-10, 10)

plt.legend()
plt.xlabel("n")
plt.ylabel("relative error")
plt.tight_layout()
#plt.savefig("relerr_simple.eps")
plt.show()
