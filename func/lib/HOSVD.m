function [Core, U, R] = HOSVD(X, R)
% performing Higth order SVD
dim        = length(size(X));
sizeX      = size(X);
U          = cell(dim,1);
Core       = X;
if nargin<2
    R = sizeX;
end

for n = 1:dim
    Xn = reshape(permute(X, [n 1:n-1 n+1:dim]),size(X,n),[]);
    if size(Xn,1)<size(Xn,2)
        XXT   = Xn*Xn';  
        if sizeX(n) <= R(n)
            [temp , ~, ~] = svd(XXT);
        else
            eigsopts.disp = 0;
            [temp , ~] = eigs(XXT, R(n) , 'LM', eigsopts);
        end
        U{n} = temp;
        Core = ModalProduct(X, U{n}, n, 'compress');
    else
        [temp , ~, ~] = svd(Xn,'econ');
        U{n} = temp(:,1:R(n));
        Core = ModalProduct(X, U{n}, n, 'compress');
    end
end

end