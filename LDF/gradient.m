function [ gradient ] = gradient( atest,Y,count )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

gradient =zeros(3,1);
for i=1:3
    for j= [count]
        if(atest(1,j)<=0)
          
            gradient(i,1)=gradient(i,1) + Y(i,j);
        end
    end
    gradient = -gradient;
end


end

