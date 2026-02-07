function [qIS,qIS_adapted,qMS_est,qVI_est] = allignDataE(qIS,qVM)
%Asks for a measured drifting qIS, reference qVI
%returns alligned drifting qIS and alligned reference qVI

A = zeros(4,4);
for i=1:size(qVM,2)
    A = A + (quatL(qVM(:,i))'*quatR(qIS(:,i)));
end

%% Define misalignment quaternions 
[U,S,V] = svd(A)
qMS_est = U(:,1);
qVI_est = V(:,1);

clc;
disp('qMS');
disp(qMS_est');
disp('qVI');
disp(qVI_est');

%% Allign  data
for i=1:size(qVM,2)
    % qVMqMS = qVIqIS
    qVS_1(:,i) =  quatmultiply(qVM(:,i)',qMS_est');
    qVS_2(:,i) =  quatmultiply(qVI_est',(qIS(:,i)'));
    
    qDelta(:,i) = quatmultiply(qVS_1(:,i)',quatconj(qVS_2(:,i)'));
    eulDelta(:,i) = quat2eul(qDelta(:,i)','zyx')*(180/pi);
    
    % (qVI*qVM)qMS = qIS
    qIS_adapted(:,i) = quatmultiply(quatmultiply(quatconj(qVI_est'),qVM(:,i)'), qMS_est');
    deltaQ(:,i) = quatmultiply(qIS_adapted(:,i)',quatconj(qIS(:,i)'));
    
    %Error between two systems in euler angels
    eulError(:,i) = quat2eul(deltaQ(:,i)','zyx')*(180/pi);
end
figure;
plot(eulError');
%RMS
disp('RMS roll pitch yaw'); 
disp(rms(eulError'));

end



