function X = InverseMDT(X_MDT,S) 
% Inverse MDT tensor data
% {\mathcal{H}_{\tau}^{-1}\left(\mathcal{X}_{H}\right)=\operatorname{unfold}_{
% (\boldsymbol{I}, \boldsymbol{\tau})}\left(\mathcal{X}_{H}\right) \times_{1} 
% \boldsymbol{S}_{1}^{\dagger} \cdots \times_{N} \boldsymbol{S}_{N}^{\dagger}}
% \boldsymbol{S}^{\dagger}:=\left(\boldsymbol{S}^{\bar{T}} \boldsymbol{S}\right)^{-1} \boldsymbol{S}^{T}
  
  N = length(S);
  size_h_tensor = zeros(1,N);
  for n = 1: N
      size_h_tensor(n) = size(S{n},1); 
      S{n} = pinv(full(S{n}));
  end
  X = reshape(full(X_MDT), size_h_tensor);
  
  X = ModalProduct_All(X,S,'decompress'); 

end