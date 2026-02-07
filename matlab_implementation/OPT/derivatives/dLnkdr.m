function der = dLnkdr(K,R,n)
% dLnk: Returns derivative of the acceleration based link between two
% sensors with respect to the sensor-joint center position vector. 

    %Senosr 1 must get a 'minus -' !!!
%     der = R*(K + (crossM(n)*K));
    der = R*K;
  
end


