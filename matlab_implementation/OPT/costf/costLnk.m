function epsilon = costLnk(q_lin_s1,q_lin_s2,C1,C2,cov_lnk)
%Returns a covariance weigthed error for the dynamics
global N;
epsilon = zeros(N,3);

    for i = 1:N
       %TODO link both segments via acceleration constraint.
       Rn1 = quat2matrix(q_lin_s1(i,:));
       Rn2 = quat2matrix(q_lin_s2(i,:));

       e(i,:) = (Rn1*C1(:,i)) - (Rn2*C2(:,i));
       epsilon(i,:) = (cov_lnk^-0.5)*e(i,:)';
      
    end

   
end
