function [MDT_X, S] = MDT(X,tau,S)
% claculates the Multi-way embedding with delay/shift along the time/space axe 
% \mathcal{H}_{\tau}(\mathcal{X})=\operatorname{fold}_{(I, \boldsymbol{\tau})}\left(\mathcal{X} \times_{1} 
% \boldsymbol{S}_{1} \cdots \times_{N} \boldsymbol{S}_{N}\right)
% Reference: "Missing Slice Recovery for Tensors Using a Low-rank Model in Embedded Space" (Tatsuya Yokota)
  N = ndims(X);
  sizes = size(X);
  
  if isempty(S)
      for n = 1:N
          S{n} = make_duplication_matrix(sizes(n),tau(n)); 
      end
      Hcol = sizes - tau + 1;
      sz = [tau; Hcol];
  else
      sz = [size(X,1) tau size(X,2) - tau + 1]; % Given mode transform
  end
  size_h_tensor = sz(:);    
  MDT_X = ModalProduct_All(full(X),S,'decompress'); 

  MDT_X = reshape(full(MDT_X),size_h_tensor');

end

function S = make_duplication_matrix(I,tau) % duplication operator
% Duplication matrix S (tau*(I-tau+1),T) of Hankel matrix (tau,(T-tau+1))
% length of series I
% embedding transform tau

  H = hankel(1:tau,tau:I); % h = hankel(r,c) where h(i,j) = p(i+j-1), p = [r c(2:end)]
  T = numel(H);
  index = [(1:T)' H(:)] - 1;
  ind = 1 + index(:,1) + T*index(:,2);
  S = sparse(T,I);
  S(ind) = 1; 
  
end
