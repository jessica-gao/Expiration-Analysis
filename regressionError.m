function [ cost, grad ] = regressionError(theta, netconfig, data, target)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient

% Extract out the "stack"
stack = params2stack(theta, netconfig);

% You will need to compute the following gradients
stackgrad = cell(size(stack));

%cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);


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
%

depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1, 1);
a{1} = data;

for layer = 1:depth
  z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
  a{layer+1} = sigmoid(z{layer+1});
end
% size(a{end})
% size(data)
cost = (0.5/M)*sum(sum((a{end}-target).^2)); 

d = cell(depth+1);
d{depth+1} = (a{depth+1}-target).*a{depth+1}.*(1-a{depth+1});
for layer = (depth:-1:2)
  d{layer} = (stack{layer}.w' * d{layer+1}) .* a{layer} .* (1-a{layer}); %当z=w*a时，z对w的导数是a的转制
end

for layer = (depth:-1:1)
  stackgrad{layer}.w = (1/M) * d{layer+1} * a{layer}';
  stackgrad{layer}.b = (1/M) * sum(d{layer+1}, 2);
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = stack2params(stackgrad);

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end