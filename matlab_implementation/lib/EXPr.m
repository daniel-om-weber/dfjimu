function [R] = EXPr(v)
%EXPR returns the DCM R for a orientation deviation parameterization
nv = norm(v);
vX = crossM(v/nv);
R = eye(3) + (sin(nv)*vX) + ((1-cos(nv))*vX^2);                %48b
% R=quat2matrix(EXPq(v));
end
