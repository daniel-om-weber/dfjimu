function K = calculateK(w,wd)
%CALCULATEK Summary of this function goes here
%   Detailed explanation goes here

global N;
K = zeros(3,3,N);
    for i = 1:N
        K(:,:,i) = (crossM(w(:,i)))^2 + crossM(wd(:,i));
    end
end

