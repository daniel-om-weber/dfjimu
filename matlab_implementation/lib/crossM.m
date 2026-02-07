function [qx] = crossM(q)
%crossM calculates the cross product matrix of a quaternion or vector
%1) A quaternion (can be a row and a column) and has the format q=[w v], with
% v = 1x3 vector
%2) A vector can be a row or a column 

if(size(q,2)==3)
    q = [0 q];
end
if(size(q,1)==3)
    q = [0 q'];
end

qx = zeros(3,3);
qx(1,2) = -q(4);
qx(1,3) = q(3);

qx(2,1) = q(4);
qx(2,3) = -q(2);

qx(3,1) = -q(3);
qx(3,2) = q(2);

end

