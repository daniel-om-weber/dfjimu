function [C,D] = calcAccatCenter(yg,dyg,ya,r)
% Returns estimate of the accelration at joint center 
    C = zeros(3,size(yg,2));
    D = zeros(3,size(yg,2));
    K = calculateK(yg,dyg);
    
    for i=1:size(yg,2)
        D(:,i) = K(:,:,i)*r';
        C(:,i) = ya(:,i) - K(:,:,i)*r';
 
%         C(:,i) = ya(:,i) + cross(dyg(:,i),p)' + cross(yg(:,i),cross(yg(:,i),p))';
    end

end

