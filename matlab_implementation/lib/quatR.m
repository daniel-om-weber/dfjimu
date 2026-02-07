function [qR] = quatR(q)
%Calculates Left quaternion multiplication of quaternion q
% 4x4 matrix
%Page 150; 

    %When a vector is given, the quaternion representation is used
    if size(q,2)==3
        q = [0 q];
    end
        qv = q(2:4);
        q0 = q(1);
        qR = zeros(4,4);

        qR(1,1) = q(1);
        qR(1,2:4) = -qv;
        qR(2:4,1) = qv;
        qR(2:4,2:4) = q0*eye(3) - crossM(q);
end

