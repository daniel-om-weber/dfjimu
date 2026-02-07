function [X] = calcX(orientation)
% Returns X vector in navigation frame described in body frame.
% orientations must be in the format b->n
global N;
X = zeros(3,N);
    for i = 1:N
          R = quat2dcm(orientation(i,:));
          X(:,i) = R(:,1);
          Y(:,i) = R(:,2);
          Z(:,i) = R(:,3);
    end
    
end

