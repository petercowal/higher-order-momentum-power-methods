
function [v3, errs] = momentum2(A, v0, n, beta, xtrue)
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
    
    h2 = norm(v2);
    x2 = v2/h2;
    v3 = A*x2;
    errs(4) = measure_error(v3, xtrue);
    
    for i = 5:n+1
        u3 = v3 - (beta/(h2*h1))*x0;
        h3 = norm(u3);
        x3 = u3/h3;
        v4 = A*x3;
        
        v3 = v4;
        x0 = x1;
        x1 = x2;
        x2 = x3;
        h1 = h2;
        h2 = h3;
        
        errs(i) = measure_error(v3, xtrue);
    end
    v3 = v3/norm(v3);
end

