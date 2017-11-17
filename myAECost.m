function [ cost, grad ] = myAECost(theta, inputSize, hiddenSize, outputNums, netconfig, lambda, data, target)
%                                          p,inputSize,hiddenSize(end),...%%¸ÄmyAEcost¡­¡­¡­¡­¡­¡­¡­¡­¡­¡­¡­¡­¡­%%%%%%%%%%%
%                          outputNum, netconfig,lambda, trainData, trainLabels),...
%                         stackedAETheta,options)
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% outputNums:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
regTheta = theta(1:hiddenSize*outputNums+outputNums);%reshape(theta(1:hiddenSize*outputNums+outputNums), outputNums, hiddenSize)
regWeight=reshape(regTheta(1:hiddenSize*outputNums),outputNums,hiddenSize);
regBias=regTheta(hiddenSize*outputNums+1:end);

% a = size(regWeight)
% b = size(regBias)
% Extract out the "stack"
stack = params2stack(theta(hiddenSize*outputNums+outputNums+1:end), netconfig);%
% stack = stack(2:end);
% c = size(stack)
% You will need to compute the following gradients
regWeightgrad = zeros(size(regWeight));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
regBiasgrad = zeros(size(regBias));%£¿£¿

stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end
cost = 0; % You need to compute this
% You might find these variables useful
M = size(data, 2);
% groundTruth = full(sparse(labels, 1:M, 1));
%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%%% 
depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1, 1);
a{1} = data;

for layer = (1:depth)%
  z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
  a{layer+1} = sigmoid(z{layer+1});
end
% e = size(a{depth+1})
result = sigmoid(regWeight * a{depth+1}+repmat(regBias,1,M));%
% n = size(result)
% p = size(regWeight)
% M = bsxfun(@minus, M, max(M));
% p = bsxfun(@rdivide, exp(M), sum(exp(M)));
% 
% cost = -1/outputNums * groundTruth(:)' * log(p(:)) + lambda/2 * sum(softmaxTheta(:) .^ 2);
% softmaxThetaGrad = -1/outputNums * (groundTruth - p) * a{depth+1}' + lambda * softmaxTheta;

%%
% M=sigmoid(weight*data+rempmat(bias,numCases));
cost = (0.5/M)*sum(sum((result-target).^2))+(0.5*lambda)*sum(sum(regTheta(:).^ 2));
%the following isn't finished
% thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta;
regWeightgrad =(1/M)*(result-target) .* result .* (1-result) * a{depth+1}' + lambda*regWeight;
regBiasgrad =(1/M)*sum((result-target).* result .* (1-result) , 2);
%%
d = cell(depth+1,1);%dÊÇÊ²Ã´£¿%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d{depth+1} = -regWeight' * ((result-target) .* result .* (1-result)) .* a{depth+1} .* (1-a{depth+1});

for layer = (depth:-1:2)
  d{layer} = (stack{layer}.w' * d{layer+1}) .* a{layer} .* (1-a{layer});
end

for layer = (depth:-1:1)
  stackgrad{layer}.w = (1/M) * d{layer+1} * a{layer}';
  stackgrad{layer}.b = (1/M) * sum(d{layer+1}, 2);
end
% -------------------------------------------------------------------------
regThetaGrad = [regWeightgrad(:);regBiasgrad(:)];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Roll gradient vector
grad = [regThetaGrad(:) ; stack2params(stackgrad)];

end
% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end