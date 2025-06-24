function [v3, errs] = momentum_dynamic(A, v0, n, xtrue)
    errs = zeros(n+1,1);
    errs(1) = measure_error(v0, xtrue);
    
    h0 = norm(v0);
    x0 = v0/h0;
    v1 = A*x0;
    errs(2) = measure_error(v1, xtrue);
    
    h1 = norm(v1);
    x1 = v1/h1;
    v2 = A*x1;
    nu1 = dot(v2,x1);
    d1 = norm(v2 - nu1*x1);
    errs(3) = measure_error(v2, xtrue);
    
    h2 = norm(v2);
    x2 = v2/h2;
    v3 = A*x2;
    nu2 = dot(v3,x2);
    d2 = norm(v3 - nu2*x2);
    rho1 = min(d2/d1, 1);
    r2 = 1/(1+(rho1-1)^2);
    errs(4) = measure_error(v3, xtrue);
    
    for i = 5:n+1
        beta = 4*(nu2*r2)^3/27;
        u3 = v3 - (beta/(h2*h1))*x0;
        h3 = norm(u3);
        x3 = u3/h3;
        v4 = A*x3;
        nu3 = dot(v4,x3);
        d3 = norm(v4 - nu3*x3);
        rho2 = min(d3/d2, 1);
        r3 = 1/(1+(rho2-1)^2);
        
        v3 = v4;
        x0 = x1;
        x1 = x2;
        x2 = x3;
        h1 = h2;
        h2 = h3;
        d2 = d3;
        nu2 = nu3;
        r2 = r3;
        
        errs(i) = measure_error(v3, xtrue);
    end
    v3 = v3/norm(v3);
end
