function [wM,wM_2] = calcRelw(qRel)
% Approximates the relative angular velocity from the relative sensor
% orientations
% In: qRel 4xN  relative orientation between two sensors
% Out: wM  Nx3  relative angular velocity
global T
    for i = 2:size(qRel,2)
        qdM(i,:) = (qRel(:,i)' - qRel(:,i-1)')/T;         
        
        wM_ = 2*quatmultiply(quatconj(qRel(:,i)'),qdM(i,:));  
        wM(i,:) = wM_(2:4);
        
%         % qt+1 = quatmultpl(qt, x % dynamic model 
%         qd = quatmultiply(qRel(:,i-1)', qRel(:,i)')/T; 
% %         % logq 
%         wM_2(i,:) = 2*LOGq(quatmultiply(quatconj(qRel(:,i)'),qd));
        
    end
end

