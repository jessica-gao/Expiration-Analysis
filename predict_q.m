load('saves/nets.mat');
load('DEHP.mat','ps', 'maxt2');

T1=60
t1=72;
ph=0.06;
T2=55;
t2=1000;

t2=t2/maxt2;
te=[T1; t1; ph; T2; t2; 0];
te=mapminmax('apply',te,ps);
te(6,:)=[];
result=myAEPredict(stackedAEOptTheta, netconfig, te);
result=result*ps.xrange(6)+ps.xmin(6);
disp(result);
