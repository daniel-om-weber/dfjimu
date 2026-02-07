%% Global variables
global Fs; global N;
global T;

%% Covariance matrices 
% TODO For this example the same for sensor 1 and sensor 2
% cov_w = eye(3)*1e-02;   
cov_w1 = cov_w(1:3,1:3);
cov_w2 = cov_w(4:6,4:6);
cov_lnk = eye(3);       % As described in 3.6
cov_a = eye(3)*1e-01;   % NOT USED 

%% Initial orientation 
% f_init = @(q)initialOrientation(q,acc,mag);
% q_1 = fminsearch(f_init,[1 0 0 0]);             
% q_1 = q_1./norm(q_1);
q_1 = [1 0 0 0];
disp(['Initial orientation: ', num2str(q_1)]);

% Intitial linearisation points
% q_lin_s1 = ones(size(gyr,2),4).*[1 0 0 0];     %Initial estimates [1 0 0 0]   
% q_lin_s2 = ones(size(gyr,2),4).*[1 0 0 0]; 
% q_lin_s2(1,:)=[0.9962 -0.0872 0 0];


% [gyr_adapted, bias_S1] = removeBias(gyr,500);
% [gyr_2adapted, bias_S2] = removeBias(gyr_2,500);
% 
% q_lin_s1 = integrateGyr(gyr_adapted',q_1);  %Initial estimates by strapdown integration of angular velocity    
% q_lin_s2 = integrateGyr(gyr_2adapted',q_1); 


% Initialize sensors have same orientation [1 0 0 0]
q_lin_s1 = integrateGyr(gyr',q_1);  %Initial estimates by strapdown integration of angular velocity    
q_lin_s2 = integrateGyr(gyr_2',q_1); 

% initialization for 4 different applied errors about vertical
% q_lin_s1 = integrateGyr(gyr',q1init);  %Initial estimates by strapdown integration of angular velocity    
% q_lin_s2 = integrateGyr(gyr_2',q2init); 

% Angular distance for strap-down integration (before applying the filter)
% for t = 1:size(q_lin_s1,1)
%     % Error relative (s1s2) orientation <Body frame>
%     rel_hat_strapd(t,:) = quatmultiply(quatconj(q_lin_s1(t,:)),q_lin_s2(t,:));
%     rel_hatEUL_strapd(t,:) = quat2eul(rel_hat_strapd(t,:))*(180/pi);
%     angular_dist_strapd(t)= angDist(qREF(:,t),rel_hat_strapd(t,:)');
% end

% q_lin_s1 = integrateGyr(gyr,q_1);

%% Gaus-Newton implementation - sensor orientation 
%length of the trial

for k = 1:10                      
    disp(['Start iteration: ', num2str(k)]);
    
    n = zeros(2*N,3);              %Initial start estimate n
    
    %% Weighted erors for current liniarization points
    %Sensor 1
    epsilon_i_s1 = costInit(n(1,:),q_lin_s1(1,:),q_1,cov_i);        
    epsilon_w_s1 = costMotion(n(1:N,:),gyr,q_lin_s1,cov_w1);
%     epsilon_a_s1 = costAcc(acc,n(1:N,:),q_lin_s1,cov_a);
    %Sensor 2
    epsilon_i_s2 = costInit(n(N+1,:),q_lin_s2(1,:),q_1,cov_i);
    epsilon_w_s2 = costMotion(n(N+1:2*N,:),gyr_2,q_lin_s2,cov_w2);
%     epsilon_a_s2 = costAcc(acc_2,n(N+1:2*N,:),q_lin_s2,cov_a);
    
    %TODO Link between Sensor 1 & Sensor 2
%     epsilon_lnk_ = costLnk(q_lin_s1,q_lin_s2,C1_r,C2_r,cov_lnk);
    epsilon_lnk = costLnk(q_lin_s1,q_lin_s2,C1,C2,cov_lnk);
%     epsilon_lnk(:,3) = epsilon_lnk_(:,3);
        %Order 
        %Sensor 1
        epsilon_w_s1 = epsilon_w_s1'; epsilon_w_s1 = epsilon_w_s1(:);
%         epsilon_a_s1 = epsilon_a_s1';epsilon_a_s1 = epsilon_a_s1(:);
%         epsilon_1 = zeros((size(epsilon_w_s1,1) + size(epsilon_a_s1,1)),1); %epsilon_w has a spot free for the epsilon_i
        epsilon_w_s1(1:3) = epsilon_i_s1;
        epsilon_1 = epsilon_w_s1;
        
        %Sensor 2 
        epsilon_w_s2 = epsilon_w_s2'; epsilon_w_s2 = epsilon_w_s2(:);
%         epsilon_a_s2 = epsilon_a_s2';epsilon_a_s2 = epsilon_a_s2(:);
%         epsilon_2 = zeros((size(epsilon_w_s2,1) + size(epsilon_a_s2,1)),1); %epsilon_w has a spot free for the epsilon_i
        epsilon_w_s2(1:3) = epsilon_i_s2;
        epsilon_2 = epsilon_w_s2;
%         Insert constraint 
%         A = [putZerosIn(epsilon_2,6,3)' [0 0 0]];
%         B = [[0 0 0 0 0 0] putZerosIn(epsilon_lnk,3,6)'];
%         epsilon_2 = A+B;
%         
        % Constraint under 
        epsilon_lnk = epsilon_lnk'; epsilon_lnk = epsilon_lnk(:);epsilon_lnk = epsilon_lnk';
        
%         epsilon = [epsilon_1 epsilon_2];
        epsilon = [epsilon_1' epsilon_2' epsilon_lnk];
    
    %% Jacobian
    %Jacobian Sensor 1: (2*3N)x3N = 2400x1200
    J_S1 = calcJac2(size(epsilon_1,1),N,n(1:N),q_lin_s1,q_1,cov_i,cov_w1,cov_a);
    %Jacobian Sensor 2: (3*3N)x3N = 3600*1200)
    J_S2 = calcJac2(size(epsilon_2,1),N,n(N+1:2*N),q_lin_s2,q_1,cov_i,cov_w2,cov_a);
%     J_S2 = calcJac_2(size(epsilon_2,2),N,n(N+1:2*N),q_lin_s1,q_lin_s2,q_1,cov_i,cov_w,cov_a);
    J_S1S2 = calcJac_Link(size(epsilon_lnk,2),(size(J_S1,2)+size(J_S2,2)),q_lin_s1,q_lin_s2,C1,C2,cov_lnk);

    %Fill Jacobian
    J = sparse((size(epsilon_1,1) + size(epsilon_2,1) + size(epsilon_lnk,2)),(2*3*N));
    J(1:size(J_S1,1),1:size(J_S1,2)) = J_S1;
    J((size(J_S1,1)+1):size(J_S1,1)+size(J_S2,1),(size(J_S1,2)+1):(size(J_S1,2)+size(J_S2,2))) = J_S2;   
%     J((size(J_S1,1)+1):end,1:size(J_S1,2)) = J_S1S2;          % combined with second sensor
    J((size(J_S1,1)+size(J_S2,1)+1):end,1:end) = J_S1S2;            % Block under everything
    
    %% Gradient & Hessian
    epsilon = epsilon'; %conversion of a matrix with vectors to one coloumn
    %Visualize summed error
    e_out(k)=(epsilon'*epsilon);     
%     figure(112); plot(e_out);

    G = J'*epsilon(:);                                        
    H = J'*J;
    
    %% Apply update            
    n = -(H\G);     %beta = 1 (line search when needed)
%     figure;plot(n);
    
    %% Update linearization point (with new n)  
    q_lin_s1 = update_linPoints(q_lin_s1,vec2mat(n(1:N*3),3));                          % Sensor 1
    q_lin_s2 = update_linPoints(q_lin_s2,vec2mat(n((N*3)+1:2*(N*3)),3));                          % Sensor 2
    
    %% *** Stopcriteria ***
    % when difference in linearisation points is small, 
    % or rotation vector 'n' is small (which describes deviation in orientation)> needed for itterations 
end
disp(['Mean of n (rotation vectors): ', num2str(norm(mean(n)))]);

% for t=1:size(q_lin_s1,1)
% %Error in absolute orientation S1
% qDelta(t,:) =  quatmultiply(q_lin_s1(t,:),quatconj(q_ref1(t,:)));
% qDeltaEuler(t,:) = quat2eul(qDelta(t,:))*(180/pi);
% %Error in absolute orientation S2
% qDelta2(t,:) =  quatmultiply(q_lin_s2(t,:),quatconj(q_ref2(t,:)));
% qDeltaEuler2(t,:) = quat2eul(qDelta2(t,:))*(180/pi);
% 
% %Error relative (s1s2) orientation <Body frame>
% relRef(t,:) = quatmultiply(quatconj(q_ref1(t,:)),q_ref2(t,:));
% rel_hat(t,:) = quatmultiply(quatconj(q_lin_s1(t,:)),q_lin_s2(t,:));
% qDeltaRel(t,:) =  quatmultiply(relRef(t,:),quatconj(rel_hat(t,:)));
% qDeltaRelEuler(t,:) = quat2eul(qDeltaRel(t,:))*(180/pi);
% %Error relative (s1s2) orientation <naviagation frame>
% relRefn(t,:) = quatmultiply((q_ref1(t,:)),quatconj(q_ref2(t,:)));
% rel_hatn(t,:) = quatmultiply((q_lin_s1(t,:)),quatconj(q_lin_s2(t,:)));
% qDeltaReln(t,:) =  quatmultiply(relRefn(t,:),quatconj(rel_hatn(t,:)));
% qDeltaRelnEuler(t,:) = quat2eul(qDeltaReln(t,:))*(180/pi);
% 
% end
% 
% 
% figure(12);
% subplot(2,1,1);
% plot(qDeltaEuler);
% subplot(2,1,2);
% plot(qDeltaEuler2);
% title('Absolute EULER orientation errors S1, S2');
% 
% figure(13);
% plot(qDeltaRelEuler);
% title('Relative EULER orientation errors S1-S2');
% 
% disp('RMSE absolute orientations');
% rms(qDeltaEuler)
% rms(qDeltaEuler2)
% disp('RMSE relative orientations');
% rms(qDeltaRelEuler)
% 
