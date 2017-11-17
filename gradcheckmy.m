%% CS294A/CS294W Programming Assignment Starter Code

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  programming assignment. You will need to complete the code in sampleIMAGES.m,
%  sparseAutoencoderCost.m and computeNumericalGradient.m.
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file.
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to
%  change the parameters below.
clear all;
clc;
height=4;
width=4;
inputSize = height * width; %输入的大小
% featureNumber=11;
outputNum = 5;
hiddenSize=[10 7];
hiddenSize=[inputSize hiddenSize];
% hiddenSize2=[4 2];
sparsityParam = [0.1 0.1 0.1 0.1];   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                       %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = [1 1 1 1];              % weight of sparsity penalty term      
%alafa = 1;
%%======================================================================

% units=rand(15,30);
data = rand(16,60);
% weight.w=rand(10,15);
% weight.b=rand(10,1);
stack{1}.w=rand(5,7);
stack{1}.b=rand(5,1);
stack{2}.w=rand(10,16);
stack{2}.b=rand(10,1);
stack{3}.w=rand(7,10);
stack{3}.b=rand(7,1);

% features=rand(11,30);
labels=rand(60,5);
% labels=zeros(60,10);
% for j = 1:60
%     for i=1:10
%         labels(i)=mod(i,3)+1;
%     end
% end 
% weight=rand(9,15);
[theta, netconfig] = stack2params(stack);

[cost,grad]=myAECost(theta, inputSize, hiddenSize(end), outputNum, netconfig, lambda, data, labels');
%softmaxCostToInput(units(:),15,30,weight,labels,9);
% (theta, inputSize, hiddenSize, outputNums, netconfig, lambda, data, target)
%[ cost,grad ] = featureCost( units(:), 15 ,30 , weight ,stack,6,features,0.1,1 );

%%======================================================================
%% STEP 3: Gradient Checking
%
% Hint: If you are debugging your code, performing gradient checking on smaller models
% and smaller training sets (e.g., using only 10 training examples and 1-2 hidden
% units) may speed things up.

% First, lets make sure your numerical gradient computation is correct for a
% simple function.  After you have implemented computeNumericalGradient.m,
% run the following:
checkNumericalGradient();

% Now we can use it to check your cost function and derivative calculations
% for the sparse autoencoder.
numgrad = computeNumericalGradient( @(p)myAECost(p, inputSize, hiddenSize(end), outputNum, netconfig, lambda, data, labels'), theta);

% Use this to visually compare the gradients side by side
format long,disp([numgrad grad]);

% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
% usually less than 1e-9.



