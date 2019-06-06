function [ countofcolumns ] = atestnegative( atest )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
count =0; 
    for i=1:length(atest)
       
        if(atest(i)<= 0)
            count=count+1;
        end
    end
    
    countofcolumns=zeros(1,count);
    k=1;
   
    
    for i=1:length(atest)
       
        if(atest(i)<= 0)
            countofcolumns(k)=i;
            k=k+1;
        end
    end



end

