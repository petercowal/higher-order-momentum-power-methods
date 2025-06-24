clear all; close all; clc;

N = 4;
A = [1.01, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, -1/3; 0, 0, 1/3, 0];
U1 = zeros(N,1);
U1(1) = 1;

iter = 200;

beta = 1/4;
beta2 = 4/27;

spectral_gap = 0.01;

disp('A=');
disp(A);

xinit = rand(N,1) + 1i*rand(N,1);
disp('xinit=');
disp(xinit);

[x1, errs1] = powermethod(A, xinit, iter, U1);
disp('power method err=');
disp(errs1(end));

[x2, errs2] = momentum(A, xinit, iter, beta, U1);
disp('momentum method err=');
disp(errs2(end));

[x3, errs3] = momentum2(A, xinit, iter, beta2, U1);
disp('order 2 momentum method err=');
disp(errs3(end));

[x4, errs4] = momentum_dynamic(A, xinit, iter, U1);
disp('order 2 dynamic momentum method err=');
disp(errs4(end));

iters = 0:iter;

figure;
semilogy(iters, errs1,'DisplayName','power method');
hold on;
semilogy(iters, errs2,'DisplayName','momentum');
semilogy(iters, errs3,'DisplayName','order 2 momentum');
semilogy(iters, errs4,'DisplayName','order 2 dyn momentum');

asympt = exp(-iters*sqrt(spectral_gap));
semilogy(iters, 1.2*asympt*errs3(101)/asympt(101), '--', 'DisplayName','theory order 2 momentum');

ylim([1e-10, 10]);

legend;
xlabel('n');
ylabel('relative error');
hold off;

