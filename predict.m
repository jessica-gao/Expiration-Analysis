load('saves/nets.mat');
load('DEHP.mat','ps', 'maxt2');

index=4;
T1=80;
t1=72;
ph=0.06;
T2=40;
t2=100;
threshold=1.5;
resolution=100;
 
data=[T1; t1; ph; T2; t2; threshold];
data=mapminmax('apply',data,ps);
data(5)=t2/maxt2;
te=repmat(data,[1,resolution+1]);
te(index,:)=(0:resolution)/resolution;
te=te(1:5,:);
result=myAEPredict(stackedAEOptTheta, netconfig, te);
result=result*ps.xrange(6)+ps.xmin(6);
figure;
if index==5
    x=(0:resolution)/resolution*maxt2;
else
    x=(0:resolution)/resolution*ps.xrange(index)+ps.xmin(index);
end
plot(x,result);
hold on;
plot(x,ones(1,resolution+1)*threshold);
switch index
    case 1
        text='老化温度';
    case 2
        text='老化时间';
    case 3
        text='极性';
    case 4
        text='迁移温度';
    case 5
        text='迁移时间';
end    

xlabel(text);
ylabel('迁移量');
title(['迁移量随', text, '变化曲线图']);

