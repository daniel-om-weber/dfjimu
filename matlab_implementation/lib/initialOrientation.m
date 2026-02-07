function q_init = initialOrientation(acc,magn)
% Returns intial orientation estimate based on first (100 samples) 
% acc and magn measurements.
qLgn = quatL([0 0 1]);
qLmn = quatL([1 0 0]);
A = zeros(4,4);

for i = 1:300
   gb = acc(:,i)/norm(acc(:,i));
   magnN = magn(:,i)/norm(magn(:,i));
   mb = cross(gb,cross(magnN,gb));
   A = A + (-(qLgn*quatR(gb'))-(qLmn*quatR(mb')));
   
%    e = (q*A*q')^2;
%    e = norm([0 0 1] - rotV(gb,q))^2 + norm([1 0 0] - rotV(mb,q)); 
%    sum = sum + e^2;
end

%% Result is the eigenvector correspoinding to the largest eigenvalue [J.D. Hol]
% [U,S,V] = svd(A);
% q_init = U(:,1)';
[V,D] = eig(A); 
[v,p]=max(diag(D));
q_init = V(:,p)';

% q_init = V(:,1)';
% qVI_est = V(:,1);

end


