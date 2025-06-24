function err = measure_error(x1, xtrue)
    x1 = x1(:);
    xtrue = xtrue(:);
    err = norm((dot(x1,xtrue)/dot(x1,x1))*x1 - xtrue)/norm(xtrue);
end
