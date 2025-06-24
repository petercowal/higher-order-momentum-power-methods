function [v1, errs] = powermethod(A, v0, n, xtrue)
    errs = zeros(n+1,1);
    errs(1) = measure_error(v0, xtrue);
    for i = 1:n
        h0 = norm(v0);
        x0 = v0/h0;
        v1 = A*x0;
        v0 = v1;
        errs(i+1) = measure_error(v1, xtrue);
    end
    v1 = v1/norm(v1);
end
