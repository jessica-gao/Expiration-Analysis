load('saves/nets_t2.mat');
load('DEHP.mat','ps', 'maxt2');

T1=100
t1=144
ph=0.06
T2=40
q=2.4768029

te=[T1; t1; ph; T2; 0; q];
te=mapminmax('apply',te,ps);
te(5,:)=[];
result=myAEPredict(stackedAEOptTheta, netconfig, te);
result=result*maxt2;
disp(result);
