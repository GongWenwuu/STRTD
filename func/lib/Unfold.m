function X_unf = Unfold(X,mode)
% Compute the unfolding (matricization) of a tensor along a specified mode
% (size(X,n)) x (size(X,1)*...*size(X,n-1)*size(X,n+1)*...*size(X,N))
    N = ndims(X);
    X_unf = reshape(permute(X, [mode 1:mode-1 mode+1:N]),size(X,mode),[]);
end