% Mulitplicative Extended Kalman filter 
% Orientation estimation for two IMUs 
% with coupled acceleration based constraint

%% Global variables
global T;  global gn; global N;                 

%% Initial orientation and 
%Initial inforamtion
q_1 = [1 0 0 0];            % ** TODO compute better estimate
    q_lin_s1 = q_1;                % linearisation on timestap 1   
    q_lin_s2 = q_1; 

P = cov_i(1)*eye(6);                  % initial Process covariance matrix 
% Q = cov_w(1)*eye(6);                  % Process noise covariance 
Q = cov_w;                  % Process noise covariance (from staic static)

% R = eye(3);                       % Measurement noise covariance not tuned
% R = eye(3)*1e-2;                  % Measurement noise covariance  (Mostly better tuned for experimental data - Evan's paper)
% Measurement noise covariance manually tuned on static window
R = eye(3)*2*(.076^2);               % Tuned 2*variance of the acc noise in rest (.076= mean std of the noise in rest)

G = T*eye(6);          

% Initialize sensors have same orientation [1 0 0 0]
orientation_s1 = integrateGyr(gyr',q_1);  %Initial estimates by strapdown integration - needed for link measurement update to converge  
    orientation_s1 =  orientation_s1';
orientation_s2 = integrateGyr(gyr_2',q_1); 
    orientation_s2 =  orientation_s2';


% Custom initialisation
% Initializize one sensor to be 10 degrees [0.9962 -0.0872 0 0] turned around x with respect to the other
% orientation_s1 = integrateGyr(gyr',q_1);  %Initial estimates by strapdown integration - needed for link measurement update to converge  
%     orientation_s1 =  orientation_s1';
% % orientation_s2 = integrateGyr(gyr_2',[0.9962 -0.0872 0 0]); 
% orientation_s2 = integrateGyr(gyr_2',q2init);     %90 degrees turned about x
%     orientation_s2 =  orientation_s2';
    
% Initialize at true orientations + rotate sensor s2 about the global vertical, to induce a relative heading error  
% orientation_s1 = integrateGyr(gyr',q1init);  %Initial estimates by strapdown integration - needed for link measurement update to converge  
%     orientation_s1 =  orientation_s1';
% orientation_s2 = integrateGyr(gyr_2',q2init);     %90 degrees turned about x
%     orientation_s2 =  orientation_s2';


for t = 1:size(orientation_s1,2)
    % Error relative (s1s2) orientation <Body frame>
    rel_hat_strapd(t,:) = quatmultiply(quatconj(orientation_s1(:,t)'),orientation_s2(:,t)');
    rel_hatEUL_strapd(t,:) = quat2eul(rel_hat_strapd(t,:))*(180/pi);
    angular_dist_strapd(t)= angDist(qREF(:,t),rel_hat_strapd(t,:)');
end

% Wrong intialization 90 degrees between the two
% orientation_s1 = zeros(4,size(gyr,2)); %initial estimates zeros.
% orientation_s1(1,:)=0.7071;
% orientation_s1(2,:)=-0.7071;
% orientation_s2 = zeros(4,size(gyr,2)); 
% orientation_s2(1,:)=0.7071;
% orientation_s2(2,:)=0.7071;

%     orientation_s1(:,1) = q_lin_s1';
%     orientation_s2(:,1) = q_lin_s2';

%% Multiplicative EKF, with orientation deviation state
% State n = zeros(3,1) is implictily done!
for t = 2:N                             
    
    %% A) Time Update
    F(1:3,1:3) = EXPr(-T*gyr(:,t-1));
        F(4:6,4:6) = EXPr(-T*gyr_2(:,t-1));
   
%    q_lin_s1 = quatmultiply(q_lin_s1, EXPq((T/2)*gyr(:,t-1)));   
%         q_lin_s2 = quatmultiply(q_lin_s2, EXPq((T/2)*gyr_2(:,t-1)));     
   q_lin_s1 = quatmultiply(orientation_s1(:,t-1)', EXPq((T/2)*gyr(:,t-1)));  
        q_lin_s2 = quatmultiply(orientation_s2(:,t-1)', EXPq((T/2)*gyr_2(:,t-1)));
        
    P = F*P*F' + G*Q*G';            
    
    %% B) Measurement update
    Rbn_s1 = quat2matrix(quatconj(q_lin_s1)); 
        Rbn_s2 = quat2matrix(quatconj(q_lin_s2)); 
    
    % Accelerometer    
%     e(1:3,1) = acc(:,t) + Rbn_s1*gn;
%         e(4:6,1) = acc_2(:,t) + Rbn_s2*gn;
    % Link segments
    e(1:3,1) = (Rbn_s1'*C1(:,t)) - (Rbn_s2'*C2(:,t));
       
    % Accelerometer   
%     H(1:3,1:3) = -crossM(Rbn_s1*gn);
%         H(4:6,4:6) = -crossM(Rbn_s2*gn);
    % Link segments
    H(1:3,1:3) = Rbn_s1'*crossM(C1(:,t));
    H(1:3,4:6) = -Rbn_s2'*crossM(C2(:,t));
    
    S = H*P*H' + R;
    K = (P*H')/S;
    
    n = K*e;
    P_tilde = P - K*S*K';
    
    %% C) Relinearize(Reset)
    q_lin_s1 = quatmultiply(q_lin_s1,EXPq(n(1:3)/2));
    q_lin_s2 = quatmultiply(q_lin_s2,EXPq(n(4:6)/2));
    
    J(1:3,1:3) = EXPr(-n(1:3));
    J(4:6,4:6) = EXPr(-n(4:6));
    P = J*P_tilde*J';  
    
    %Output
    e_out(t) = norm(e);
    P_out(:,t) = [sqrt(P(1,1)) sqrt(P(2,2)) sqrt(P(3,3))]'*180/pi;
    orientation_s1(:,t) = q_lin_s1;
        orientation_s2(:,t) = q_lin_s2;    
    n_out(:,t) = n;

%     qDelta_s1(t,:) =  quatmultiply(orientation_s1(:,t)',quatconj(q_ref1(t,:)));
%     qDeltaEuler(t,:) = quat2eul(qDelta_s1(t,:))*(180/pi);
%         qDelta_s2(t,:) =  quatmultiply(orientation_s2(:,t)',quatconj(q_ref2(t,:)));
%         qDeltaEuler_s2(t,:) = quat2eul(qDelta_s2(t,:))*(180/pi);
%         
        
%     %Error in absolute orientation S1
%     qDelta(t,:) =  quatmultiply(orientation_s1(:,t)',quatconj(q_ref1(t,:)));
%     qDeltaEuler(t,:) = quat2eul(qDelta(t,:))*(180/pi);
%     %Error in absolute orientation S2
%     qDelta2(t,:) =  quatmultiply(orientation_s2(:,t)',quatconj(q_ref2(t,:)));
%     qDeltaEuler2(t,:) = quat2eul(qDelta2(t,:))*(180/pi);
%     
%     %Error relative (s1s2) orientation <Body frame>
%     relRef(t,:) = quatmultiply(quatconj(q_ref1(t,:)),q_ref2(t,:));
%     rel_hat(t,:) = quatmultiply(quatconj(orientation_s1(:,t)'),orientation_s2(:,t)');
%     qDeltaRel(t,:) =  quatmultiply(relRef(t,:),quatconj(rel_hat(t,:)));
%     qDeltaRelEuler(t,:) = quat2eul(qDeltaRel(t,:))*(180/pi);
%     %Error relative (s1s2) orientation <naviagation frame>
%     relRefn(t,:) = quatmultiply((q_ref1(t,:)),quatconj(q_ref2(t,:)));
%     rel_hatn(t,:) = quatmultiply((orientation_s1(:,t)'),quatconj(orientation_s2(:,t)'));
%     qDeltaReln(t,:) = quatmultiply(relRefn(t,:),quatconj(rel_hatn(t,:)));
%     qDeltaRelnEuler(t,:) = quat2eul(qDeltaReln(t,:))*(180/pi);    
end

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







