function dy = approxDerivative(y)
%Returns approximating first derivative 
% Finite difference approximation

% https://en.wikipedia.org/wiki/Numerical_differentiation 
% https://en.wikipedia.org/wiki/Five-point_stencil 
global Fs;
    % Five-point stencil implementation
    dy = zeros(1, size(y,2));
    dy(3:end-2)= (y(1:end-4) - 8*y(2:end-3) +8*y(4:end-1)-y(5:end))*(Fs/12);
    
    % Two-point implementation 
%     dy = zeros(1, size(y,2));
%     dy(3:end) = (y(3:end) - y(1:end-2))/(2/Fs);

end

%% Test function
% t = 0:0.1:10
% y = t.^2
% dy = approxDerivative(y,1/0.1)
% figure
% plot(t,y)
% hold on
% plot(t,dy)
%%
