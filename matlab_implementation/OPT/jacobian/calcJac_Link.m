function [J] = calcJac_Link(m,n,q_lin_s1,q_lin_s2,C1,C2,cov_lnk)
%Calculation of Jacobian matrix block for the other sensor link part .. 
global N;

J = sparse(m,n);

    % relative unknowns (sensor 1)
    for t=1:N
        Rn1 = quat2matrix(q_lin_s1(t,:));   %Rnb with b=1 for sensor 1
        Rn2= quat2matrix(q_lin_s2(t,:));
        
        pos_link_r = 1+(t-1)*3;
        pos_link_c = 1+(t-1)*3;
        
        %S1
        J(pos_link_r:pos_link_r+2,pos_link_c:pos_link_c+2) = ...
            (cov_lnk^-0.5)*(-dLnk(Rn1,C1(:,t)));
        
        %S2 (N positions further)
        J(pos_link_r:pos_link_r+2,(3*N)+pos_link_c:(3*N)+pos_link_c+2) = ...
            (cov_lnk^-0.5)*(dLnk(Rn2,C2(:,t)));
        

    end

end

