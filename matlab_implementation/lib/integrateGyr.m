function orientation = integrateGyr(gyr,q_1)
%INTEGRATEGYR 
% Returns qnb  ?
global T;
    orientation = zeros(size(gyr,1),4);
    orientation(1,:) = q_1;

    for i = 2:size(gyr,1)
      orientation(i,:) = quatmultiply(orientation(i-1,:),EXPq((T/2)*gyr(i,:)));
    end 
        
end

