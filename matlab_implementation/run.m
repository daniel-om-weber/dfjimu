% Script to process extra experimental datasets
% Data (Dustin Lehmann)
% Experimental set-up: two connected segments with interchangeble joint
% 1D/2D/3D
% Some arbitrary movements and durations. 
clear;clc;
addpath(genpath('lib'));addpath(genpath('data'));
addpath(genpath('OPT'));addpath(genpath('MEKF'));

%% Load data
% it=5; load(strcat('data_1D_0',num2str(it),'.mat'));     % 1D
it=7; load(strcat('data_2D_0',num2str(it),'.mat'));   % 2D
% it=5; load(strcat('data_3D_0',num2str(it),'.mat'));   % 3D

% Chaning namings and dimentions 
r1 = data.r_12';r1 = -r1;          % Position vectors (changed convention of axis direction)
r2 = data.r_21';r2 = -r2;
acc = data.sensorData(:,1:3)';     % Inertial data
gyr = data.sensorData(:,4:6)';
acc_2 = data.sensorData(:,7:9)';
gyr_2 = data.sensorData(:,10:12)';
qGS1_ref = data.ref(:,1:4);        % reference 
qGS2_ref = data.ref(:,5:8);


%% Reference 
qREF = zeros(4,size(qGS1_ref,1));
for p=1:size(qGS1_ref,1)  
    qREF(:,p) = quatmultiply(quatconj(qGS1_ref(p,:)) ,qGS2_ref(p,:));
end

%% Parameters
global Fs;Fs  = 50; global T;T = 1/Fs;  
global N; global g;  g = 9.82; global gn; gn = [0 0 -g]';
N = size(gyr,2);

% Gyro noise covariance
noise = gyr(1:3,1:40);sdNoise_gyr1 = std(noise');
noise = gyr_2(1:3,1:40);sdNoise_gyr2 = std(noise');
cov_w = eye(6);
cov_w(1,1) = sdNoise_gyr1(1); cov_w(2,2) = sdNoise_gyr1(2); cov_w(3,3) = sdNoise_gyr1(3); 
cov_w(4,4) = sdNoise_gyr2(1); cov_w(5,5) = sdNoise_gyr2(2); cov_w(6,6) = sdNoise_gyr2(3);
% intial orientation est. cov.
cov_i = eye(3)*0.35^2;

% Approximate joint center acclerations and angular accelerations
dgyr = [approxDerivative(gyr(1,:));approxDerivative(gyr(2,:));approxDerivative(gyr(3,:))];
dgyr_2 = [approxDerivative(gyr_2(1,:));approxDerivative(gyr_2(2,:));approxDerivative(gyr_2(3,:))];
[C1 D1] = calcAccatCenter(gyr,dgyr,acc,(r1)');
[C2 D2] = calcAccatCenter(gyr_2,dgyr_2,acc_2,(r2)');


%% Run filter 
% MEKF_conn;
% for t = 1:size(orientation_s1,2)
%     rel_hat(t,:) = quatmultiply(quatconj(orientation_s1(:,t)'),orientation_s2(:,t)');
%     rel_hatEUL(t,:) = quat2eul(rel_hat(t,:))*(180/pi);
% end

%% Run smoother
OPT_conn; 
for t = 1:size(q_lin_s1,1)
    rel_hat(t,:) = quatmultiply(quatconj(q_lin_s1(t,:)),q_lin_s2(t,:));
    rel_hatEUL(t,:) = quat2eul(rel_hat(t,:))*(180/pi);
end


%% calculate angular distance
for t = 1:N-1
% angular_dist(t)= angDist(qREF(:,t),rel_hat(t,:)');

% Additional time shift (improved accuracy for this specific experimental dataset)
angular_dist(t)= angDist(qREF(:,t),rel_hat(t+1,:)'); 
end

%% Plot result
figure;plot(angular_dist);
xlabel('Samples # (Fs:50Hz)');
xlabel('Angular distance (°)');

