%Load trial information
vicon = ViconNexus;
subject = vicon.GetSubjectNames{1};

global startFrame
global EndFrame
[startFrame, EndFrame] = vicon.GetTrialRegionOfInterest;

%modelOut = vicon.GetModelOutputNames('2sensor');
%(output in degrees)
% s1w =  vicon.GetModelOutput(subject,'s1w');
% s2w =  vicon.GetModelOutput(subject,'s2w');
% s1s2 =  vicon.GetModelOutput(subject,'s1$s2');
% s2s1 =  vicon.GetModelOutput(subject,'s2$s1');
% relHeading =  vicon.GetModelOutput(subject,'relHeading');

% Needs the following in ProCalc: 
% ShankThigh =  vicon.GetModelOutput(subject,'shankThigh')';
ShankThigh =  vicon.GetModelOutput(subject,'shank2Thigh')';

%% Correct for gimbal lock if needed
for t = 1:size(ShankThigh,1)
    if ShankThigh(t,1)<0
    ShankThigh_Adapted(t) = (ShankThigh(t,1))+360;
    else
    ShankThigh_Adapted(t) = ShankThigh(t,1);
    end
end
ShankThigh(:,1) = ShankThigh_Adapted;

% Convert reference to quaternions
relRef = eul2quat(ShankThigh*(pi/180),'XYZ');

%figure;plot(qTS);title('');

%qTSNew = eul2quat(ShankThigh*(pi/180),'ZYX');

%Calculate Tilt filter JKLEE (output: in B (Roll), Y (pitch)
%remove offset to be able to compare.

%Sensor 1
% figure;
% plot(-(ws1(1,startFrame:EndFrame)));
% hold on;
% plot(B*180/pi);
% title('S_1 Roll');
% legend('Vicon','IMU JK LEE 2012');
% 
% figure;
% plot((ws1(3,startFrame:EndFrame)));
% hold on;
% plot(Y*180/pi)
% title('S_1 Pitch');
% legend('Vicon','IMU JK LEE 2012');
% 
% %Sensor 2
% figure;
% plot((ws2(1,startFrame:EndFrame)));
% hold on;
% plot(B*180/pi);
% title('S_1 Roll');
% legend('Vicon','IMU JK LEE 2012');
% 
% figure;
% plot((ws2(3,startFrame:EndFrame)));
% hold on;
% plot(Y*180/pi)
% title('S_1 Pitch');
% legend('Vicon','IMU JK LEE 2012');

%After fusion data and calculating joint angles
% for t = 1:size(q_lin_s1,1)
% %Error relative (s1s2) orientation <Body frame>
% rel_hat(t,:) = quatmultiply(quatconj(q_lin_s1(t,:)),q_lin_s2(t,:));
% end
% [S1a,S1ar] = allignData(rel_hat',relRef');
% 






