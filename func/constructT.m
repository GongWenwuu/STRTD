function T = constructT(sz)

    c = zeros(1,sz-1);c(1) = 1;
    r = zeros(1,sz);r(1) = 1;r(2) = -1;
    T = toeplitz(c,r);
end