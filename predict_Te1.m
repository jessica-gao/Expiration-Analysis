load('saves/nets_Te1.mat');
load('DEHP.mat','ps', 'maxt2');

t1=72;
ph=0.06
T2=40
t2=192
q=1.1218812

te=[0; t1; ph; T2; t2; q];
te=mapminmax('apply',te,ps);
te(1,:)=[];
result=myAEPredict(stackedAEOptTheta, netconfig, te);
result=result*ps.xrange(1)+ps.xmin(1);
disp(result);
