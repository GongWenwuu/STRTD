function [Z, G, U, hist] = STRTD(X,Omega,Opts) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                       Alternating proximal gradient for STRTD                                      %
%\underset{\mathcal{G}; \{\mathbf{U}_{n}\}; \mathcal{X}}{\operatorname{minimize}} \frac{1}{2} \left\| \mathcal{X} - \mathcal{G} \times_{n=1}^{N} {\mathbf{U}_{n}}\right\|^2_F 
% + \alpha\|\mathcal{G}\|_{1} + \sum_{n=1}^{K} \frac{\beta_{n}}{2} \operatorname{tr}\left(\mathbf{U}_{n}^{\mathrm{T}} \mathbf{L}_{n} \mathbf{U}_{n}\right) 
% + \sum_{n=K+1}^{N} \frac{\beta_{n}}{2} \|\mathbf{T}_{n}\mathbf{U}_{n}\|_{F}^{2} \} 
% \text{s.t.,} \mathbf{U}_{n} \in \mathbb{R}_{+}^{I_{n} \times I_{n}}, n=1, \ldots, N \ \text{and} \ \mathcal{X}_\Omega = \mathcal{X}^0_\Omega,
%                                       This code was written by Wenwu Gong (2022.09)                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(Opts,'lambda');  lambda = Opts.lambda; else; lambda = 1;   end
if isfield(Opts,'Rpara');   Rpara = Opts.Rpara;   else; Rpara = 0.01; end
if isfield(Opts,'maxit');   maxit = Opts.maxit;   else; maxit = 300;  end
if isfield(Opts,'tol');     tol = Opts.tol;       else; tol = 1e-4;   end
if isfield(Opts,'phi');     phi = Opts.phi;       else; phi = 0;      end

N = ndims(X);
Z = X.*Omega; Z(~Omega) = mean(X(Omega));
[Ginit,Uinit,~] = Initial(Z,Rpara,'lrstd');
L = cell(1,N); T = cell(1,N); beta = zeros(1,N);
if isfield(Opts, 'flag')
    for n = 1:N
        if Opts.flag(n) == 1
            if strcmp(Opts.prior, 'traffic')
                L{n} = constructT(size(Z, n));
                L{n} = L{n}'*L{n};
                beta(n) = 1/(0.2*norm(L{n}, 2));
            elseif strcmp(Opts.prior, 'fg')
                if isfield(Opts,'Xtr')
                    L = ConstructL_lrstd(Opts.Xtr, 3, 0);
                    Xmat = reshape(permute(Opts.Xtr, [n 1:n-1 n+1:N]), size(Z, n), []);
                    beta(n) = norm(Xmat, 2)/(2*norm(L{n}, 2));
                end  
            end
        elseif Opts.flag(n) == 0
            if strcmp(Opts.prior, 'traffic') 
                Teo = zeros(1,size(Ginit, n) - 3);
                T{n} = toeplitz([1 -2 1 Teo]);
                for i = 2:size(Ginit, n)-1
                    T{n}(i,i-1) = 0;
                    T{n}(i+1,i-1) = 0;
                end
                beta(n) = 1/(0.2*norm(T{n}*T{n}', 2));
            elseif strcmp(Opts.prior, 'fg')
                if isfield(Opts,'Xtr')
                    Teo = constructT(size(Ginit, n));
                    T{n} = Teo';
                    Xmat = reshape(permute(Opts.Xtr, [n 1:n-1 n+1:N]), size(Z, n), []);
                    beta(n) = norm(Xmat, 2)/(2*norm(T{n}*T{n}', 2));
                end
            end 
        else
            beta(n) = 0;
        end   
    end
else
    Opts.flag = 2.*ones(1,N);
end

obj0 = loss(X, Omega, Ginit, Uinit, L, T, beta, Opts);
Usq = cell(1,N); 
for n = 1:N
    Usq{n} = Uinit{n}'*Uinit{n};
end

t0 = 1;
G = Ginit; Gextra = Ginit; Lgnew = 1;
U = Uinit; Uextra = Uinit; LU0 = ones(N,1); LUnew = ones(N,1);
gradU = cell(N,1); wU = ones(N,1);

fprintf('Iteration:     ');
for iter = 1:maxit
    fprintf('\b\b\b\b\b%5i',iter); 
    % -- Core tensor updating --  
    Lg0 = Lgnew;
    gradG = gradientG(Gextra, U, Usq, Z);
    Lgnew = lipG(Usq, 1);
    G = sign(Gextra - gradG/Lgnew).*max(0,abs(Gextra - gradG/Lgnew) - Opts.alpha/(lambda*Lgnew));
    for n = 1:N     
        % -- Factor matrices updating --
        gradU{n} = gradientU(Uextra, U, Usq, G, Z, L{n}, T{n}, beta(n), n, Opts.flag(n)); 
        LU0(n) = LUnew(n);
        LUnew(n) = lipU(Usq, G, L{n}, T{n}, beta(n), n, Opts.flag(n), 1);
        U{n} = max(0,Uextra{n} - gradU{n}/(lambda*LUnew(n)));   
        Usq{n} = U{n}'*U{n};  
    end  

    Z_pre = Z;
    Z_new = ModalProduct_All(G,U,'decompress');
    Z(~Omega) = Z_new(~Omega);
    Z(Omega) = X(Omega) + phi*(Z(Omega) - Z_new(Omega));

    % -- diagnostics and reporting --
    objk = loss(X, Omega, G, U, L, T, beta, Opts); 
    hist.obj(iter) = objk;
    relchange = norm(Z_new(:)-Z_pre(:))/norm(Z_pre(:));
    hist.err(1,iter) = relchange;
    relerr = abs(objk - obj0)/(obj0 + 1);
    hist.err(2,iter) = relerr;
    if isfield(Opts,'Xtr')
        rmse = sqrt((1/length(nonzeros(~Omega)))*norm(Opts.Xtr(~Omega)-Z(~Omega),2)^2);
        hist.rmse(iter) = rmse;
        rse = norm(Opts.Xtr(~Omega)-Z(~Omega))/norm(Opts.Xtr(:));
        hist.rse(iter) = rse;
        nmae = norm(Opts.Xtr(~Omega)-Z(~Omega),1)/norm(Opts.Xtr(~Omega),1);
        hist.nmae(iter) = nmae;
    end
    
    % -- stopping checks and correction --
    if relchange < tol 
        break;
    end
    
    % -- extrapolation --      
    t = (1+sqrt(1+4*t0^2))/2;
    if objk >= obj0
        G = Ginit;
        U = Uinit;
    else
        w = (t0-1)/t;
        wG = min([w,0.999*sqrt(Lg0/Lgnew)]);
        Gextra = G + wG*(G - Ginit); 
        for n = 1:N
            wU(n) = min([w,0.9999*sqrt(LU0(n)/LUnew(n))]);
            Uextra{n} = U{n}+wU(n)*(U{n}-Uinit{n});
        end
        Ginit = G; Uinit = U; t0 = t; obj0 = objk; 
    end
end
Z = Z_new;
Z(Omega) = X(Omega);

end