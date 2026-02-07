function [gyr_adapted, bias] = removeBias(gyr,samples)
%Estimate bias from gyroscope in quasistatic time moment.
%Return bias-removed gyroscope samples. 
    
    bias = mean(gyr(:,1:samples)');
    gyr_adapted = gyr - bias';
end

