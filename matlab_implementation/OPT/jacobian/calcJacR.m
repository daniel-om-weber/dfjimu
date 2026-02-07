function [J] = calcJacR(K1,K2,q_lin_s1,q_lin_s2,n1,n2,cov_r)
%Calculation of Jacobian matrix block for the other sensor link part 
global N;

J = sparse(3*N,6);

    for t=1:N
        
        Rn1 = quat2matrix(q_lin_s1(t,:));   %Rnb with b=1 for sensor 1
        Rn2= quat2matrix(q_lin_s2(t,:));
        
        row = 1+(t-1)*3;
        
        %R1
        J(row:row+2,1:3) = (0.05^-0.5)*(-dLnkdr(K1(:,:,t),Rn1,n1(t,:)));
        %R2
        J(row:row+2,4:6) = (0.05^-0.5)*(dLnkdr(K2(:,:,t),Rn2,n2(t,:)));
        
    end

end

