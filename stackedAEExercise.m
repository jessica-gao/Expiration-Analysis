%% CS294A/CS294W Stacked Autoencoder Exercise
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.
clc;clearvars;

% addpath mnistdata\;
% addpath G:\tianliG\特征提取方法\GLCM ;
% data creat

load('DEHP.mat', 'data');  


features = data;
features(:,3)=[];
labels = data(:,3);

% features=data(:,1:5);
% labels = data(:,6);

DISPLAY = false;
inputSize = 5; %输入的大小
outputNum = 1; %输出个数

hiddenSize=[10 20 50 100 60 30 12 4];
sparsityParam = [0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01];   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                       %  in the lecture notes). 
lambda = 3e-6;         % weight decay parameter       
beta = 1e-4;              % weight of sparsity penalty term       

%%======================================================================
%% 打乱数据
features = features';
index = randperm(length(labels));
features = features(:,index);
labels = labels(index);
clear index;

%% --训练数据
% 百分之80%训练
percentile=floor(length(labels));
trainData = features(:,1:percentile);
trainLabels = labels(1:percentile)';
testData = trainData; % features(:,percentile+1:end);
testLabels = trainLabels; % labels(percentile+1:end)';

%%======================================================================
%% STEP 2: Train sparse autoencoder
%  This trains sparse autoencoder on the unlabelled training data

%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.
%  Randomly initialize the parameters

addpath minFunc/;
options = struct;
options.Method = 'lbfgs';
options.maxIter = 300; %最大迭代次数
options.display = 'false';

layers=length(hiddenSize);
saeOptTheta=cell(layers,1);
hiddenSize=[inputSize hiddenSize];
saeFeatures=trainData;
%%
for i=1:layers
    saeTheta = initializeParameters(hiddenSize(i+1), hiddenSize(i));
    [weight, ~] =  minFunc(@(p)sparseAutoencoderCost(p,...
    hiddenSize(i),hiddenSize(i+1),lambda,sparsityParam(i),beta,saeFeatures),saeTheta,options);%训练出第一层网络的参数
    saeOptTheta{i}=weight;
    save(['saves/step' num2str(i) '.mat'], 'weight');

    if DISPLAY
      W = reshape(weight(1:hiddenSize(i+1) * hiddenSize(i)), hiddenSize(i+1), hiddenSize(i));
      display_network(W');
    end
    [saeFeatures] = feedForwardAutoencoder(weight, hiddenSize(i+1), ...
    hiddenSize(i), saeFeatures);
end

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);
% topSaeTheta = cell(layers,1);

topSaeTheta = myreg_Train(hiddenSize(end),outputNum,saeFeatures,trainLabels,options);

save(['saves/step' num2str(layers+1) '.mat'], 'topSaeTheta');
% -------------------------------------------------------------------------

%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(layers+1,1);
for j=1:layers
    stack{j}.w = reshape(saeOptTheta{j}(1:hiddenSize(j+1)*hiddenSize(j)), ...
                     hiddenSize(j+1), hiddenSize(j));
    stack{j}.b = saeOptTheta{j}(2*hiddenSize(j+1)*hiddenSize(j)+1:2*hiddenSize(j+1)*hiddenSize(j)+hiddenSize(j+1));
end
stack{layers+1}.w=reshape(topSaeTheta(1:hiddenSize(end)*outputNum),outputNum,hiddenSize(end));
stack{layers+1}.b=topSaeTheta(hiddenSize(end)*outputNum+1:end);

% Initialize the parameters for the deep model
[stackedAETheta, netconfig] = stack2params(stack);
%stackedAETheta = [topSaeTheta ; stackparams ];%stackedAETheta是个向量，为整个网络的参数，包括分类器那部分，且分类器那部分的参数放前面

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%
[stackedAEOptTheta, cost] =  minFunc(@(p)regressionError(p,netconfig,trainData,trainLabels),stackedAETheta,options);%训练出第一层网络的参数
%                 

save(['saves/step' num2str(layers+2) '.mat'], 'stackedAEOptTheta');

save('saves/nets_ph.mat', 'stackedAEOptTheta', 'netconfig');
% figure;
% if DISPLAY
%   optStack = params2stack(stackedAEOptTheta(hiddenSizeL2*outputNum+1:end), netconfig);
%   W11 = optStack{1}.w;
%   W12 = optStack{2}.w;
%   % TODO(zellyn): figure out how to display a 2-level network
% %    display_network(log(1 ./ (1-W11')) * W12');
% end
% -------------------------------------------------------------------------

%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
% testData = loadMNISTImages('t10k-images.idx3-ubyte');
% testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
%% 测试数据
 

% testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = myAEPredict(stackedAETheta, netconfig, testData);
p = 1/length(testLabels);
erro=p*sum(sum((pred-testLabels).^2));
fprintf('Before Finetuning Test Error: %0.5f\n', erro);
erro2 = p*(sum(abs(pred-testLabels)));
fprintf('Before Finetuning Test Error2: %0.5f\n', erro2);

[pred] = myAEPredict(stackedAEOptTheta, netconfig, testData);

erro=p*sum(sum((pred-testLabels).^2));
fprintf('After Finetuning Test Error: %0.5f\n', erro);
erro2 = p*(sum(abs(pred-testLabels)));
fprintf('After Finetuning Test Error2: %0.5f\n', erro2);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)

% x = 1:1:100;
% testL = testLabels(:,1:100);
% pre = pred(:,1:100);
% 
% p = plot(x,testL,'.-r',x,pre,'.-b');
% title('autoencoder回归预测结果曲线');% xlabel('x');
% ylabel('y');
% toc;
