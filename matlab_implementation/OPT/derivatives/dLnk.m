function der = dLnk(R,C)
% dLnk: Returns derivative of the acceleration based link between two
% sensors. 
    %Senosr 1 must get a 'minus -' !!!
    der = R*crossM(C);
  
end


