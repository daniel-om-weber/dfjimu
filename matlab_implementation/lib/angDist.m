function [ang_d] = angDist(s,r)
% Calculate the angular distance ang_d between two quaternions s, r 
% Hartley et al. (2013)

%     dotp = dot(quatconj(s'),r');
%     ang_d = 2*acos(abs(dotp))*180/pi;
%     ang_d = real(ang_d);  % in case of very small diferences
   
%     ang_d = dist(r',s)*180/pi;

%     ang_d = acos(2*abs(dot(s,r')) -1)*180/pi;


%%% Karsten

    quat_Diff = quatmultiply(quatinv(r'), s');

    for i=1:length(quat_Diff(:,1))
        if quat_Diff(i,1) < 0
            quat_Diff(i,:) = -quat_Diff(i,:);
        end
    end

    ang_d = rad2deg(2*real(acos(quat_Diff(:,1))));


end

