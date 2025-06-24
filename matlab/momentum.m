function [v2, errs] = momentum(A, v0, n, beta, xtrue)
    errs = zeros(n+1,1);
    errs(1) = measure_error(v0, xtrue);
    
    h0 = norm(v0);
    x0 = v0/h0;
    v1 = A*x0;
    errs(2) = measure_error(v1, xtrue);
    
    h1 = norm(v1);
    x1 = v1/h1;
    v2 = A*x1;
    errs(3) = measure_error(v2, xtrue);
    
    for i = 4:n+1
        u2 = v2 - (beta/h1)*x0;
        h2 = norm(u2);
        x2 = u2/h2;
        v3 = A*x2;
        
        v2 = v3;
        x0 = x1;
        x1 = x2;
        h0 = h1;
        h1 = h2;
        
        errs(i) = measure_error(v2, xtrue);
    end
    v2 = v2/norm(v2);
end
