function X = Fold(X_unf,mode,sz)
% Inverse operation of matrizicing, i.e. reconstructs the multi-way array from it's matriziced version.
% sz is vector containing original dimensions
    N = length(sz);
    if mode == 1
        perm = 1:N;
    else
        perm = [2:mode 1 mode+1:N];
    end
    X = permute(reshape(X_unf,sz([mode 1:mode-1 mode+1:N])),perm);
end