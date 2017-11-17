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
        text='�ϻ��¶�';
    case 2
        text='�ϻ�ʱ��';
    case 3
        text='����';
    case 4
        text='Ǩ���¶�';
    case 5
        text='Ǩ��ʱ��';
end    

xlabel(text);
ylabel('Ǩ����');
title(['Ǩ������', text, '�仯����ͼ']);

