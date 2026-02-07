function [M] = dlogdq()
    %returns 3*4 matrix
    M = zeros(4,3);
    M(2:4,:) = eye(3);
    M = M';
end

