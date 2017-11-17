function [cost, grad] = myreg_Cost(theta, outputNums, inputSize, data, target)
% outputNums - the number of classes 
% myreg_Cost(p, ...%%改mycost…………………………………%%%%%%%%%%%
%                                    outputNums, inputSize, lambda, ...
%                                    inputData, labels), ...                                   
%                               theta, options);
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta

numCases = size(data, 2);%输入样本的个数

%%
% a = size(weight);100 1
% bias=theta(outputNums*inputSize+1:end);
% c = repmat(bias,numCases);
% size(c)
%%
weight=theta(1:outputNums*inputSize);
% size(weight)
bias=theta(outputNums*inputSize+1:end);
weight = reshape(weight, outputNums, inputSize);%将输入的参数列向量变成一个矩阵
%%
% size(theta)
% a = size(weight)
% b = size(bias)
% c = repmat(bias,1,numCases);
% d = size(c)
% e = repmat(bias',1,numCases);
% f = size(e)
%%
cost = 0;

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

%%
M=sigmoid(weight*data+repmat(bias,1,numCases));
cost = (0.5/numCases)*sum(sum((M-target).^2)); %+(0.5*lambda)*sum(sum(weight.^2));
%the following isn't finished
% thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta;
weightgrad =(1/numCases)*(M-target).*M.*(1-M)*data'; % + lambda*weight;
biasgrad =(1/numCases)*sum((M-target).*M.*(1-M),2);


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [weightgrad(:);biasgrad(:)];
end
