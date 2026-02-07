function sum = costJointPosition(x,gyro_data1,gyro_data2,acc_data1,acc_data2,derigyro_data1,derigyro_data2,j1,j2)
sum=0;
sum2=0;
sum1=0;

for i = 1:size(gyro_data1,1)
    gamma1 = cross(gyro_data1(i,:),cross(gyro_data1(i,:),x(1:3))) + cross(derigyro_data1(i,:),x(1:3));
    gamma2 = cross(gyro_data2(i,:),cross(gyro_data2(i,:),x(4:6))) + cross(derigyro_data2(i,:),x(4:6));
    
  
    %first penalty based on absolute deviation of the acceleration
    error = norm(acc_data1(i,:)+gamma1) - norm(acc_data2(i,:)+gamma2);%vem relatieve acc plus acc door verplaatsting moet gelijk zijn aan elkaar

    w1=norm(gyro_data1(i,:))+norm(gyro_data2(i,:));
    w2=0.1*norm(derigyro_data1(i,:)+derigyro_data2(i,:));
    sum1 = sum1 + error*error*(w1+w2); % weight added when the measurement data deviates from gravity then the results are interesting, otherwise, they contain noise
    
    error = abs(a1hkas-a2hkas)+abs(a1opas-a2opas);
    sum2 = sum2 + error*error;
    
end
sum=sum1+0.02*sum2;