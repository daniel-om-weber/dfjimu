function epsilon = costAcc(acc,n ,q_lin, cov_a)
%Returns a covariance weigthed error for acc gravity readings
% cov_a is function of the accelerometer norm
global gn;
global N;

epsilon = zeros(size(acc,1),3);

    for i = 1:N
        
       Rbn = quat2matrix(quatconj(q_lin(i,:))); 
       e = acc(:,i) + Rbn*gn;
       
       epsilon(i,:) = (cov_a(1)^-0.5)*e';     %weighted error
    end
end


