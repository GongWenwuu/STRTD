function para = initial_para(maxit, Rpara, alpha)

para.maxit = maxit;             % maximum iteration number of APG
para.Rpara = Rpara;             % Initial G size
para.alpha = alpha;             % Core shresholding, Low-rank

end