function [q] = EXPq(v)
%Expects row vector

%When coloumn vector is given -> make row vector. 
if(size(v,1)==3)
    v=v';
end

q = [cos(norm(v)) (v/(norm(v)+realmin))*sin(norm(v))];                      % 48a
% q = [1 v];                                                                %approximation 52a

end

