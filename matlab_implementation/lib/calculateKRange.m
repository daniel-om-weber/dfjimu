function K = calculateKRange(w,wd)
%CALCULATEK Summary of this function goes here
%   Detailed explanation goes here

N = size(w,2);
K = zeros(3,3,N);
    for i = 1:N
        K(:,:,i) = (crossM(w(:,i)))^2 + crossM(wd(:,i));
    end
end

