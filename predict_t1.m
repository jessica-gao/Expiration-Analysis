load('saves/nets_t1.mat');
load('DEHP.mat','ps', 'maxt2');

T1=60
ph=0.06
T2=40;
t2=24
q=0.3283458

te=[T1; 0; ph; T2; t2; q];
te=mapminmax('apply',te,ps);
te(2,:)=[];
result=myAEPredict(stackedAEOptTheta, netconfig, te);
result=result*ps.xrange(2)+ps.xmin(2);
disp(result);
