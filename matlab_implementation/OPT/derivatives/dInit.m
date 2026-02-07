function der = dInit(q_1,q_lin)
% dInit: Returns derivative of the costInit wrt state at time step 1.
    der = (dlogdq*quatL(quatmultiply(quatconj(q_1),q_lin))*dexpndn);
end

