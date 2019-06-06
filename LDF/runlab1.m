%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 1, Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Attribute Information for IRIS data:
%    1. sepal length in cm
%    2. sepal width in cm
%    3. petal length in cm
%    4. petal width in cm

%    class label/numeric label: 
%       -- Iris Setosa / 1 
%       -- Iris Versicolour / 2
%       -- Iris Virginica / 3


%% this script will run lab1 experiments..
close all;
clear
load irisdata.mat

%% extract unique labels (class names)
labels = unique(irisdata_labels);

%% generate numeric labels
numericLabels = zeros(size(irisdata_features,1),1);
for i = 1:size(labels,1)
    numericLabels(find(strcmp(labels{i},irisdata_labels)),:)= i;
end

%% feature distribution of x1 for two classes
%figure

    
%subplot(1,2,1), hist(irisdata_features(find(numericLabels(:)==1),2),100), title('Iris Setosa, sepal width (cm)');

%subplot(1,2,2), hist(irisdata_features(find(numericLabels(:)==2),2),100); title('Iris Veriscolour, sepal width (cm)');

%figure

%subplot(1,2,1), hist(irisdata_features(find(numericLabels(:)==1),1),100), title('Iris Setosa, sepal length (cm)');
%subplot(1,2,2), hist(irisdata_features(find(numericLabels(:)==2),1),100); title('Iris Veriscolour, sepal length (cm)');
    

%  figure
% % 
%  plot(irisdata_features(find(numericLabels(:)==1),2),irisdata_features(find(numericLabels(:)==1),3),'rs'); title('x_2 vs x_3');
%  hold on;
%  plot(irisdata_features(find(numericLabels(:)==2),2),irisdata_features(find(numericLabels(:)==2),3),'k.');
% axis([1.9 4.5 0.5 5.5]);

    

%% build training data set for two class comparison
% merge feature samples with numeric labels for two class comparison (Iris
% Setosa vs. Iris Veriscolour
% trainingSet = [irisdata_features(1:100,:) numericLabels(1:100,1) ];
% 
% 
% %% Lab1 experiments (include here)
% %%QUESTION 1
% %x = [3:0.01:4];
% max = 0;
% maxx = 0;
% maxl = 0;
% maxxl = 0;
% %for value in x
% for x = [0:0.05:4]
%     a = length(find(irisdata_features(find(numericLabels(:)==1),2) > x));
%     b = length(find(irisdata_features(find(numericLabels(:)==2),2) < x));
%     total = a + b;
%     if total > max;
%         max = total;
%         maxx = x;
%     end;
% end;
% 
% error = (1 - (max / (100))) * 100
% dividingsepalwidth = maxx
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%QUESTION 3
% lab1length(1,3.3,irisdata_features)
% lab1length(1,4.4,irisdata_features)
% lab1length(1,5.0,irisdata_features)
% lab1length(1,5.7,irisdata_features)
% lab1length(1,6.3,irisdata_features)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% m11 = 5.006;  % mean of the class conditional density p(x/w1)= [p(w1/x)*p(x)]/p(w1)
% std11 =0.3525; % Standard deviation of the class conditional density p(x/w1)
% 
% m12 = 5.9360; % mean of the class conditional density p(x/w2)
% std12= 0.5162; % Standard deviation of the class conditional density p(x/w2)
% 
% w=[3:.1:9];
% norm=(1/((std11)*sqrt(2*pi)))* (exp((-0.5)*((w-m11)/(std11)).^2));
% figure;
% plot(w,norm)
% 
% hold on;
% 
% d=[3:.1:9];
% norm=normpdf(d,m12,std12);
% plot(d,norm)
% title('Normpdf for Class1 and Class2 Length');
% legend('class1','class2')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% m11 = 3.418;  % mean of the class conditional density p(x/w1)= [p(w1/x)*p(x)]/p(w1)
% std11 =0.3810; % Standard deviation of the class conditional density p(x/w1)
% 
% m12 = 2.77; % mean of the class conditional density p(x/w2)
% std12= 0.3138; % Standard deviation of the class conditional density p(x/w2)
% 
% w=[1:.1:5];
% norm=(1/((std11)*sqrt(2*pi)))* (exp((-0.5)*((w-m11)/(std11)).^2));
% figure;
% plot(w,norm)
% 
% hold on;
% 
% d=[1:.1:5];
% norm=normpdf(d,m12,std12);
% plot(d,norm)
% title('Normpdf for Class1 and Class2 Width');
% legend('class1','class2')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %test for width
% lab1length(2,1.7,irisdata_features);
% lab1length(2,2.3,irisdata_features);
% lab1length(2,3.05,irisdata_features);
% lab1length(2,3.7,irisdata_features);
% lab1length(2,4.3,irisdata_features);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %FIND THRESHOLD FOR LENGTH
% 
% for xl = [0.05:0.1:7.5]
%     al = length(find(irisdata_features(find(numericLabels(:)==1),1) < xl));
%     bl = length(find(irisdata_features(find(numericLabels(:)==2),1) > xl));
%     totall = al + bl;
%     if totall > maxl
%         maxl = totall ;     
%         maxxl=xl;
%    
%     end;
% end;
% error = (1 - (maxl / (100))) * 100
% dividingsepall = maxxl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

valueClassA(1:50,1:2) = irisdata_features(1:50,2:3);
valueClassB(1:50,1:2) = irisdata_features(51:100,2:3);
valueClassC(1:50,1:2) = irisdata_features(101:150,2:3);

%% Augmentation and Normalization of 30 percent sample AB
%30 percent Width Column 1 x2, Column 2 Petal Length x3
valueClassAB30(1:15,2) = valueClassA(1:15,1);%width
valueClassAB30(16:30,2) = valueClassB(1:15,1);%width
valueClassAB30(1:15,3) = valueClassA(1:15,2); %% petal length
valueClassAB30(16:30,3) = valueClassB(1:15,2);%% petal length
valueClassAB30(1:15,1) = 1; %% classifier
valueClassAB30(16:30,1) = 2;%% classifier

Ytest=valueClassAB30.';
Y=valueClassAB30.';

Y=augnandnorm(Y);

nk=0.01;
a=[0; 0; 1];
theta=0;

%Perception Multiplication
atest=a.'*Y; %for the initial values perception
countofcolumns=atestnegative(atest);%find which columns are misclassified
%this counts the misclassifications and returns the 3x1 matrix of sums
gradientx=gradient(atest,Y,countofcolumns);
p=1;
o=1;
 
for k= 1:299
    
    a=a-nk*gradientx;
    
    atest=a.'*Y;
    
    countofcolumns=atestnegative(atest);%find which columns are misclassified
    
    %this counts the misclassifications and returns the 3x1 matrix of sums
     if((length(countofcolumns) == 0))%store the value of atest it is solution
           
               fprintf("done ");
            value(1:3,p)=a(1:3,1);
            p=p+1;
            kcount(1)=k+1;
            break;
            
     else
           fprintf("1");
         valuebad1(1:3,o)=a(1:3,1);
         o=o+1;
        gradientx=gradient(atest,Y,countofcolumns);
        
     end
  
   
end %found weight vector with 30% samples

solu30(1:3,1)=value(1:3,1)

%% 70 percent AB
%70 percent Width Column 1 x2, Column 2 Petal Length x3
valueClassAB70(1:35,2) = valueClassA(16:50,1);%width
valueClassAB70(36:70,2) = valueClassB(16:50,1);%width
valueClassAB70(1:35,3) = valueClassA(16:50,2); %% petal length
valueClassAB70(36:70,3) = valueClassB(16:50,2);%% petal length
valueClassAB70(1:35,1) = 1; %% classifier
valueClassAB70(36:70,1) = 2;%% classifier

Ytest=valueClassAB70.';
Y=valueClassAB70.';

Y=augnandnorm70(Y);

nk=0.01;
a=[0; 0; 1];
theta=0;

%Perception Multiplication
atest=a.'*Y; %for the initial values perception
countofcolumns=atestnegative(atest);%find which columns are misclassified
%this counts the misclassifications and returns the 3x1 matrix of sums
gradientx=gradient(atest,Y,countofcolumns);
p=1;
o=1;
k=0;
 
for k= 1:299
    
    a=a-nk*gradientx;
    
    atest=a.'*Y;
    
    countofcolumns=atestnegative(atest);%find which columns are misclassified
    
    %this counts the misclassifications and returns the 3x1 matrix of sums
     if((length(countofcolumns) == 0))%store the value of atest it is solution
            value2(1:3,p)=a(1:3,1);
            p=p+1;
              kcount(2)=k+1;
            break;
            
     else
         valuebad2(1:3,o)=a(1:3,1);
         o=o+1;
        gradientx=gradient(atest,Y,countofcolumns);
        
     end
  
   
end %found weight vector with 30% samples
solu70(1:3,1)=value2(1:3,1)
   
%% 30 BC
%30 Width column 1 x2, column 2 petal length x3
valueClassBC30(1:15,2) = valueClassB(1:15,1);%width
valueClassBC30(16:30,2) = valueClassC(1:15,1);%width
valueClassBC30(1:15,3) = valueClassB(1:15,2); %% petal length
valueClassBC30(16:30,3) = valueClassC(1:15,2);%% petal length
valueClassBC30(1:15,1) = 1; %% classifier
valueClassBC30(16:30,1) = 2;%% classifier

Ytest=valueClassBC30.';
Y=valueClassBC30.';

Y=augnandnorm(Y);

nk=0.01;
a=[0; 0; 1];
theta=0;

%Perception Multiplication
atest=a.'*Y; %for the initial values perception
countofcolumns=atestnegative(atest);%find which columns are misclassified
%this counts the misclassifications and returns the 3x1 matrix of sums
gradientx=gradient(atest,Y,countofcolumns);
p=1;
o=1;

for k= 1:299
    
    a=a-nk*gradientx;
    
    atest=a.'*Y;
    
    countofcolumns=atestnegative(atest);%find which columns are misclassified
    
    %this counts the misclassifications and returns the 3x1 matrix of sums
     if((length(countofcolumns) == 0))%store the value of atest it is solution
            fprintf("Value Solution");
            value3(1:3,p)=a(1:3,1);
            p=p+1;
            
     else
        valuebad3(1:3,o)=a(1:3,1);
        o=o+1;
        gradientx=gradient(atest,Y,countofcolumns);
        
     end
  
   
end %found weight vector with 30% samples
%solu(1:3,1)=value3(1:3,1)
%% 70 BC

%70 percent Width Column 1 x2, Column 2 Petal Length x3
valueClassBC70(1:35,2) = valueClassB(16:50,1);%width
valueClassBC70(36:70,2) = valueClassC(16:50,1);%width
valueClassBC70(1:35,3) = valueClassB(16:50,2); %% petal length
valueClassBC70(36:70,3) = valueClassC(16:50,2);%% petal length
valueClassBC70(1:35,1) = 1; %% classifier
valueClassBC70(36:70,1) = 2;%% classifier


Ytest=valueClassBC70.';
Y=valueClassBC70.';

Y=augnandnorm70(Y);

nk=0.01;
a=[0; 0; 1];
theta=0;

%Perception Multiplication
atest=a.'*Y; %for the initial values perception
countofcolumns=atestnegative(atest);%find which columns are misclassified
%this counts the misclassifications and returns the 3x1 matrix of sums
gradientx=gradient(atest,Y,countofcolumns);
p=1;
o=1;

for k= 1:299
    
    a=a-nk*gradientx;
    
    atest=a.'*Y;
    
    countofcolumns=atestnegative(atest);%find which columns are misclassified
    
    %this counts the misclassifications and returns the 3x1 matrix of sums
     if((length(countofcolumns) == 0))%store the value of atest it is solution
           
       
            value4(1:3,p)=a(1:3,1);
            p=p+1;
            
            
     else
          valuebad4(1:3,o)=a(1:3,1);
         o=o+1;
        gradientx=gradient(atest,Y,countofcolumns);
        
     end
  
   
end %found weight vector with 30% samples
%solu(1:3,1)=value4(1:3,1)
%% Graph for class 1 and 2 x2 and x3 30% LINE


figure(1)
% plot(irisdata_features(1:15,2),irisdata_features(1:15,3),'rs'); 
% hold on;
% plot(irisdata_features(51:65,2),irisdata_features(51:65,3),'k.');
% hold on;

plot(irisdata_features(16:50,2),irisdata_features(16:50,3),'rs'); % 70%data set 
hold on;
plot(irisdata_features(66:100,2),irisdata_features(66:100,3),'k.');
hold on;

syms x y;
var=value(1:3,1).'*[1; x; y] == 0;
    ySol=isolate(var,y);
    ezplot(ySol);axis([0 4.5 0 5.5]); title('AB 30% Data x_2 vs x_3 Optimal nk=0.01'); xlabel('x2'),ylabel('x3') ;
    
figure(2)
% plot(irisdata_features(1:15,2),irisdata_features(1:15,3),'rs'); 
% hold on;
% plot(irisdata_features(51:65,2),irisdata_features(51:65,3),'k.');
% hold on;

plot(irisdata_features(16:50,2),irisdata_features(16:50,3),'rs'); % 70%data set 
hold on;
plot(irisdata_features(66:100,2),irisdata_features(66:100,3),'k.');
hold on;

 

for i= 1:2
    var=valuebad1(1:3,i).'*[1; x; y] == 0;
    ySol=isolate(var,y);
    ezplot(ySol);
    hold on;
end
title('AB 30% Data x_2 vs x_3 nk=0.01');xlabel('x2'),ylabel('x3') ;

axis([0 4.5 0 5.5]);

%% Graph for class 1 and 2 x2 and x3 70% LINE
figure(3)
% plot(irisdata_features(16:50,2),irisdata_features(16:50,3),'rs'); 
% hold on;
% plot(irisdata_features(66:100,2),irisdata_features(66:100,3),'k.');
% hold on;

plot(irisdata_features(1:15,2),irisdata_features(1:15,3),'rs'); 
hold on;
plot(irisdata_features(51:65,2),irisdata_features(51:65,3),'k.');
hold on;

syms x y;
var=value2(1:3,1).'*[1; x; y] == 0;
    ySol70=isolate(var,y);
    ezplot(ySol70);axis([0 4.5 0 5.5]); title('AB 70% Data x_2 vs x_3 Optimal nk=0.01'); xlabel('x2'),ylabel('x3') ;
    
figure(4)
% plot(irisdata_features(16:50,2),irisdata_features(16:50,3),'rs'); 
% hold on;
% plot(irisdata_features(66:100,2),irisdata_features(66:100,3),'k.');
% hold on;
 
plot(irisdata_features(1:15,2),irisdata_features(1:15,3),'rs'); 
hold on;
plot(irisdata_features(51:65,2),irisdata_features(51:65,3),'k.');
hold on;

for i= 1
    var=valuebad2(1:3,i).'*[1; x; y] == 0;
    ySol70=isolate(var,y);
    ezplot(ySol70);
    hold on;
end
title('AB 70% Data x_2 vs x_3 nk=0.01');xlabel('x2'),ylabel('x3') ;

axis([0 4.5 0 5.5]);

%% nk values 0.02 

%70 percent Width Column 1 x2, Column 2 Petal Length x3
valueClassAB70(1:35,2) = valueClassA(16:50,1);%width
valueClassAB70(36:70,2) = valueClassB(16:50,1);%width
valueClassAB70(1:35,3) = valueClassA(16:50,2); %% petal length
valueClassAB70(36:70,3) = valueClassB(16:50,2);%% petal length
valueClassAB70(1:35,1) = 1; %% classifier
valueClassAB70(36:70,1) = 2;%% classifier

Ytest=valueClassAB70.';
Y=valueClassAB70.';

Y=augnandnorm70(Y);

nk=0.02;
a=[0; 0; 1];
theta=0;

%Perception Multiplication
atest=a.'*Y; %for the initial values perception
countofcolumns=atestnegative(atest);%find which columns are misclassified
%this counts the misclassifications and returns the 3x1 matrix of sums
gradientx=gradient(atest,Y,countofcolumns);
p=1;
o=1;
 
for k= 1:299
    
    a=a-nk*gradientx;
    
    atest=a.'*Y;
    
    countofcolumns=atestnegative(atest);%find which columns are misclassified
    
    %this counts the misclassifications and returns the 3x1 matrix of sums
     if((length(countofcolumns) == 0))%store the value of atest it is solution
            value5(1:3,p)=a(1:3,1);
            p=p+1;
            kcount(3)=k+1;
            break;
            
     else
          valuebad5(1:3,o)=a(1:3,1);
         o=o+1;
        gradientx=gradient(atest,Y,countofcolumns);
        
     end
  
   
end %found weight vector with 30% samples
solu70nk2(1:3,1)=value5(1:3,1)
%%

%%GRAPH nk=0.02
%% Graph for class 1 and 2 x2 and x3 70% LINE
figure(5)
% plot(irisdata_features(find(numericLabels(:)==1),2),irisdata_features(find(numericLabels(:)==1),3),'rs'); 
% hold on;
% plot(irisdata_features(find(numericLabels(:)==2),2),irisdata_features(find(numericLabels(:)==2),3),'k.');
% hold on;
plot(irisdata_features(1:15,2),irisdata_features(1:15,3),'rs'); 
hold on;
plot(irisdata_features(51:65,2),irisdata_features(51:65,3),'k.');
hold on;

syms x y;
var=value5(1:3,1).'*[1; x; y] == 0;
    ySol70=isolate(var,y);
    ezplot(ySol70);axis([0 4.5 0 5.5]); title('70% Data x_2 vs x_3 Optimal nk=0.02'); xlabel('x2'),ylabel('x3') ;
    
figure(6)
% plot(irisdata_features(find(numericLabels(:)==1),2),irisdata_features(find(numericLabels(:)==1),3),'rs'); 
% hold on;
% plot(irisdata_features(find(numericLabels(:)==2),2),irisdata_features(find(numericLabels(:)==2),3),'k.');
% hold on;
 plot(irisdata_features(1:15,2),irisdata_features(1:15,3),'rs'); 
hold on;
plot(irisdata_features(51:65,2),irisdata_features(51:65,3),'k.');
hold on;

for i= 1
    var=value5(1:3,i).'*[1; x; y] == 0;
    ySol70=isolate(var,y);
    ezplot(ySol70);
    hold on;
end
title('70% Data x_2 vs x_3 nk=0.02');xlabel('x2'),ylabel('x3') ;

axis([0 4.5 0 5.5]);

%% nk values 0.03 

%70 percent Width Column 1 x2, Column 2 Petal Length x3
valueClassAB70(1:35,2) = valueClassA(16:50,1);%width
valueClassAB70(36:70,2) = valueClassB(16:50,1);%width
valueClassAB70(1:35,3) = valueClassA(16:50,2); %% petal length
valueClassAB70(36:70,3) = valueClassB(16:50,2);%% petal length
valueClassAB70(1:35,1) = 1; %% classifier
valueClassAB70(36:70,1) = 2;%% classifier

Ytest=valueClassAB70.';
Y=valueClassAB70.';

Y=augnandnorm70(Y);

nk=0.03;
a=[0; 0; 1];
theta=0;

%Perception Multiplication
atest=a.'*Y; %for the initial values perception
countofcolumns=atestnegative(atest);%find which columns are misclassified
%this counts the misclassifications and returns the 3x1 matrix of sums
gradientx=gradient(atest,Y,countofcolumns);
p=1;
o=1;
for k= 1:299
    
    a=a-nk*gradientx;
    
    atest=a.'*Y;
    
    countofcolumns=atestnegative(atest);%find which columns are misclassified
    
    %this counts the misclassifications and returns the 3x1 matrix of sums
     if((length(countofcolumns) == 0))%store the value of atest it is solution
            value6(1:3,p)=a(1:3,1);
            p=p+1;
            kcount(4)=k+1;
            break;
            
     else
         valuebad6(1:3,o)=a(1:3,1);
         o=o+1;
        gradientx=gradient(atest,Y,countofcolumns);
        
     end
  
   
end %found weight vector with 30% samples
solu70nk3(1:3,1)=value6(1:3,1)
%% small nk has a slow convergence, with larger nk we see a faster convergence however there
%is the possibility of oversoot and the divergence, for the values of nk
%=0.02 and 0.03 we see faster convergence 
%% 
%%GRAPH 70 nk =0.03
%% Graph for class 1 and 2 x2 and x3 70% LINE
figure(7)
% plot(irisdata_features(find(numericLabels(:)==1),2),irisdata_features(find(numericLabels(:)==1),3),'rs'); 
% hold on;
% plot(irisdata_features(find(numericLabels(:)==2),2),irisdata_features(find(numericLabels(:)==2),3),'k.');
% hold on;
 plot(irisdata_features(1:15,2),irisdata_features(1:15,3),'rs'); 
hold on;
plot(irisdata_features(51:65,2),irisdata_features(51:65,3),'k.');
hold on;

syms x y;
var=value6(1:3,1).'*[1; x; y] == 0;
    ySol70=isolate(var,y);
    ezplot(ySol70);axis([0 4.5 0 5.5]); title('70% Data x_2 vs x_3 Optimal nk=0.03'); xlabel('x2'),ylabel('x3') ;

    
figure(8)
% plot(irisdata_features(find(numericLabels(:)==1),2),irisdata_features(find(numericLabels(:)==1),3),'rs'); 
% hold on;
% plot(irisdata_features(find(numericLabels(:)==2),2),irisdata_features(find(numericLabels(:)==2),3),'k.');
% hold on;
  plot(irisdata_features(1:15,2),irisdata_features(1:15,3),'rs'); 
hold on;
plot(irisdata_features(51:65,2),irisdata_features(51:65,3),'k.');
hold on;


% for i= 290:293
%     var=valuebad6(1:3,i).'*[x; y; 1] == 0;
%     ySol70=isolate(var,y);
%     ezplot(ySol70);
%     hold on;
% end
% title('70% Data x_2 vs x_3 nk=0.03');xlabel('x2'),ylabel('x3') ;
% 
% axis([0 4.5 0 5.5]);
%%
%TEST TEST TEST
%% Augmentation and Normalization of 30 percent sample AB
%30 percent Width Column 1 x2, Column 2 Petal Length x3

%%TEST TEST TEST

%%
%% nk values 0.03 

%70 percent Width Column 1 x2, Column 2 Petal Length x3
valueClassAB70(1:35,2) = valueClassA(16:50,1);%width
valueClassAB70(36:70,2) = valueClassB(16:50,1);%width
valueClassAB70(1:35,3) = valueClassA(16:50,2); %% petal length
valueClassAB70(36:70,3) = valueClassB(16:50,2);%% petal length
valueClassAB70(1:35,1) = 1; %% classifier
valueClassAB70(36:70,1) = 2;%% classifier

Ytest=valueClassAB70.';
Y=valueClassAB70.';

Y=augnandnorm70(Y);

nk=0.01;
a=[0; 0; 2];
theta=0;

%Perception Multiplication
atest=a.'*Y; %for the initial values perception
countofcolumns=atestnegative(atest);%find which columns are misclassified
%this counts the misclassifications and returns the 3x1 matrix of sums
gradientx=gradient(atest,Y,countofcolumns);
p=1;
o=1;
 
for k= 1:299
    
    a=a-nk*gradientx;
    
    atest=a.'*Y;
    
    countofcolumns=atestnegative(atest);%find which columns are misclassified
    
    %this counts the misclassifications and returns the 3x1 matrix of sums
     if((length(countofcolumns) == 0))%store the value of atest it is solution
            value7(1:3,p)=a(1:3,1);
            p=p+1;
            kcount(5)=k+1;
            break;
            
     else
         valuebad7(1:3,o)=a(1:3,1);
         o=o+1;
        gradientx=gradient(atest,Y,countofcolumns);
        
     end
  
   
end %found weight vector with 30% samples
solu70nk1a(1:3,1)=value5(1:3,1)
%%

figure(9)
% plot(irisdata_features(find(numericLabels(:)==1),2),irisdata_features(find(numericLabels(:)==1),3),'rs'); 
% hold on;
% plot(irisdata_features(find(numericLabels(:)==2),2),irisdata_features(find(numericLabels(:)==2),3),'k.');
% hold on;
 plot(irisdata_features(1:15,2),irisdata_features(1:15,3),'rs'); 
hold on;
plot(irisdata_features(51:65,2),irisdata_features(51:65,3),'k.');
hold on;

syms x y;
var=value7(1:3,1).'*[1; x; y] == 0;
    ySol70=isolate(var,y);
    ezplot(ySol70);axis([0 4.5 0 5.5]); title('70% Data x_2 vs x_3 Optimal nk=0.03 a=[0,0,2]'); xlabel('x2'),ylabel('x3') ;

    
figure(10)
% plot(irisdata_features(find(numericLabels(:)==1),2),irisdata_features(find(numericLabels(:)==1),3),'rs'); 
% hold on;
% plot(irisdata_features(find(numericLabels(:)==2),2),irisdata_features(find(numericLabels(:)==2),3),'k.');
% hold on;
%  
 plot(irisdata_features(1:15,2),irisdata_features(1:15,3),'rs'); 
hold on;
plot(irisdata_features(51:65,2),irisdata_features(51:65,3),'k.');
hold on;
for i=1:2
    var=valuebad7(1:3,i).'*[1; x; y] == 0;
    ySol70=isolate(var,y);
    ezplot(ySol70);
    hold on;
end
title('70% Data x_2 vs x_3 nk=0.03 a=[0,0,2] error');xlabel('x2'),ylabel('x3') ;

axis([0 4.5 0 5.5]);











