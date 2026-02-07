function der = dAcc(q_lin)
% dAcc: Returns derivative of the costAcc wrt state at timestap 't'
global gn;
    der = crossM(quat2matrix(q_lin)'*gn);
end

