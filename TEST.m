dense_tensor = G;
G_Eval_STRTD = zeros(5,13);
DLP = [0.1:0.1:0.9,0.93,0.95,0.97,0.99];
for MR = 1:length(DLP)
    rng('default')
    filename=['G_STRTD_' num2str(MR) '.mat'];
    sample_ratio = 1- DLP(MR);
    sample_num = round(sample_ratio*numel(dense_tensor));
    fprintf('Sampling tensor with %4.1f%% known elements ...... \n',100*sample_ratio);
    idx = 1:numel(dense_tensor);
    idx = idx(dense_tensor(:)>0);
    mask = sort(randperm(length(idx),sample_num));
    arti_miss_idx = idx;  
    arti_miss_idx(mask) = [];  
    arti_miss_mv = dense_tensor(arti_miss_idx);
    Omega = zeros(size(dense_tensor)); Omega(mask) = 1; Omega = boolean(Omega);
    sparse_tensor = Omega.*dense_tensor;
    fprintf('Known elements / total elements: %6d/%6d.\n',sample_num,numel(dense_tensor));
    clear idx 

    t0 = tic;
    Opts = initial_para(300,size(dense_tensor),1); Opts.Xtr = dense_tensor; Opts.flag = [1,1,0]; Opts.prior = 'fg'; 
    [est_tensor, ~, ~, info] = STRTD(sparse_tensor,Omega,Opts); 
    save(filename,"Omega",'sparse_tensor','est_tensor','dense_tensor',"info","Opts")
    G_Eval_STRTD(5,MR) = toc(t0);
    rse = TensorNorm(est_tensor - dense_tensor,'fro')/TensorNorm(dense_tensor,'fro');
    nmae = norm(arti_miss_mv-est_tensor(arti_miss_idx),1) / norm(arti_miss_mv,1);
    rmse = sqrt((1/length(arti_miss_mv))*norm(arti_miss_mv-est_tensor(arti_miss_idx),2)^2);  
    mape = (100/length(arti_miss_mv))* sum(abs((arti_miss_mv-est_tensor(arti_miss_idx))./arti_miss_mv));
    G_Eval_STRTD(4,MR) = mape; G_Eval_STRTD(3,MR) = rmse; G_Eval_STRTD(1,MR) = nmae; G_Eval_STRTD(2,MR) = rse;

end