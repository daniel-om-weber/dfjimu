function [qL] = quatL(q)
%Calculates Left quaternion multiplication of quaternion q
% 4x4 matrix
%Page 150; 

    %When a vector is given, the quaternion representation is used
    if size(q,2)==3
        q = [0 q];
    end
        qv = q(2:4);
        q0 = q(1);
        qL = zeros(4,4);

        qL(1,1) = q(1);
        qL(1,2:4) = -qv;
        qL(2:4,1) = qv;
        qL(2:4,2:4) = q0*eye(3) + crossM(q);
end

