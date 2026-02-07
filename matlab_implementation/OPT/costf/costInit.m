function epsilon = costInit(n,q_lin,q_init,cov_i)
%Returns a covariance weigthed error for the dynamics
   a = EXPq(n/2);
   e = 2*LOGq(quatR(a)*quatL(quatconj(q_init))*q_lin');
   
   epsilon = (cov_i^-0.5)*e';     %weighted error
end

