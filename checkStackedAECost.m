function [] = checkStackedAECost()

% Check the gradients for the stacked autoencoder
%
% In general, we recommend that the creation of such files for checking
% gradients when you write new cost functions.
%

%% Setup random data / small model
inputSize = 4;
hiddenSize = 5;
lambda = 0.01;
sparsityParam = 0.1;                               
beta = 3;  
data   = randn(inputSize, 5);


saeTheta = initializeParameters(hiddenSize, inputSize);
[~, grad] =sparseAutoencoderCost(saeTheta,inputSize,hiddenSize,lambda,sparsityParam,beta,data);

% [cost, grad] = stackedAECost(stackedAETheta, inputSize, hiddenSize, ...
%                              numClasses, netconfig, ...
%                              lambda, data, labels);

% Check that the numerical and analytic gradients are the same
numgrad = computeNumericalGradient( @(p)sparseAutoencoderCost(p,...
    inputSize,hiddenSize,lambda,sparsityParam,beta,data), ...
                                        saeTheta);

% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
disp('Norm between numerical and analytical gradient (should be less than 1e-9)');
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

            % When you got this working, Congratulations!!! 
            
            
