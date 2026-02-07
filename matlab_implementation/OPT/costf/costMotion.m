function epsilon = costMotion(n,gyr,q_lin,cov_w)
%Returns a covariance weigthed error for the dynamics
epsilon = zeros(size(gyr,1),3);
global T; global N;

for i = 2:N
   
   %Dynamics 
   a = quatconj(EXPq(n(i-1,:)/2));
   b = quatmultiply(a,quatconj(q_lin(i-1,:)));
   c = quatmultiply(b,q_lin(i,:));
   d = quatmultiply(c,EXPq(n(i,:)/2));
   
   e = 2/T*LOGq(d)-gyr(:,i-1)';
   
   epsilon(i,:) = (cov_w^-0.5)*e';     %weighted error
end

end
