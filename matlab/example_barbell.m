clear all; close all; clc;

N = 160;
p = 1/10;

X1 = double(rand(N, N) < p);
X2 = double(rand(N, N) < p);

B = [X1, zeros(N,N); zeros(N,N), X2];

B(N, N+1) = 1;
B(N+1, N) = 1;

A = B./sum(B,1);


figure;
imagesc(A);
colorbar;

iter = 500;

[U,lmds] = eig(A);
lmds = diag(lmds);
[~, idx] = sort(abs(lmds),'descend');
lmds = lmds(idx);
U = U(:,idx);

U1 = U(:,1);

figure;

theta = linspace(0,2*pi,60);
curve = (2/3*exp(1i*theta) + 1/3*exp(-2i*theta));

plot(cos(theta),sin(theta),':','DisplayName','|z|=1');
hold on;
plot(real(curve),imag(curve),'--','DisplayName','\gamma^{(3)}');
plot(real(lmds),imag(lmds),'.','DisplayName','\lambda_n');

legend;
xlim([-1.2,1.2]);
ylim([-1.2,1.2]);
axis square;
hold off;

beta = abs(lmds(2))^2/4;
beta2 = 4*abs(lmds(2))^3/27;

spectral_gap = abs(lmds(1))/abs(lmds(2)) - 1;
fprintf('spectral gap = %f\n', spectral_gap);

xinit = rand(2*N,1) + 1i*rand(2*N,1);

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
hold all;
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
