function [ Yout ] = augnandnorm( Y )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
for i=1
      for j=1:30
    if(Y(i,j)==2)
         Y(i,j)= -1;
         Y(i+1,j)= -Y(i+1,j);
         Y(i+2,j)= -Y(i+2,j);
    end 
      end
end 

Yout=Y;

end

