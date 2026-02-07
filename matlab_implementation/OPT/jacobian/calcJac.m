function [J] = calcJac(M,N,n,q_lin,q_1,cov_i,cov_w,cov_a)
%Calculation of Jacobian matrix block for one sensor (No link)

J = sparse(M,N);
% C = sparse(M,N);      % debug
% figure;

% relative unknowns
for t=1:1:N
    %% Initial 
    if t==1
        J(1:3,1:3) = (cov_i^-0.5)*dInit(q_1,q_lin(1,:));
%         C(1:3,1:3) = 2;
    end
    
    %% e_acc -> n(t)
    pos_acc_r = 4+(t-1)*6; 
    pos_acc_c = 1+(t-1)*3;
    J(pos_acc_r:pos_acc_r+2,pos_acc_c:pos_acc_c+2) = ...
        (cov_a^-0.5)*dAcc(q_lin(t,:));
%   C(pos_acc_r:pos_acc_r+2,pos_acc_c:pos_acc_c+2) = 3;
    
    %% Dynamics
    if(t>1)
        %%  ew -> n(t-1)
        pos_wt1_r = 7+(t-2)*6;
        pos_wt1_c = 1 + (t-2)*3;
        J(pos_wt1_r:pos_wt1_r+2,pos_wt1_c:pos_wt1_c+2) = ...
            (cov_w^-0.5)*dMotiontm1(q_lin(t,:),q_lin(t-1,:)); 
%       C(pos_wt1_r:pos_wt1_r+2,pos_wt1_c:pos_wt1_c+2) = 4;  
        
        %%  ew -> n(t)
        J(pos_wt1_r:pos_wt1_r+2,pos_wt1_c+3:pos_wt1_c+3+2) = ...
            (cov_w^-0.5)*dMotion(q_lin(t,:),q_lin(t-1,:));
%       C(pos_wt1_r:pos_wt1_r+2,pos_wt1_c+3:pos_wt1_c+3+2) = 5;
    end 
    
%     spy(C==2,'r'); hold on; spy(C==3,'g');hold on; spy(C==4,'b');hold on; spy(C==5,'c');
end
%Extra parameters
end

