function [ cost,grad ] = featureCost(zz,weight,features,sparsityParam,beta)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
[m,n]=size(features);
w=weight.w;
b=weight.b;
zz=reshape(zz,size(w,2),n);
units=sigmoid(zz);
w=w(1:m,:);
b=b(1:m);
z=w*units+repmat(b,1,n);
f=sigmoid(z);
rho = (1/n).*sum(units,2);%求出平均值向量
Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+(1-sparsityParam).*log((1-sparsityParam)./(1-rho)));
cost=(0.5/n)*sum(sum((f-features).^2))+beta*Jsparse;
d=(f-features).*sigmoidInv(z);
sterm = beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));
grad=(1/n).*(w'*d+repmat(sterm,1,n));
grad=grad.*sigmoidInv(zz);
grad=grad(:);

end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function sigmInv = sigmoidInv(x)

    sigmInv = sigmoid(x).*(1-sigmoid(x));
end