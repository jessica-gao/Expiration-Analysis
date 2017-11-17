load('saves/nets_ph.mat');
load('DEHP.mat','ps', 'maxt2');

T1=50;
t1=100;
T2=25;
t=2
q=0.35


te=[T1; t1; 0; T2; t2; q];
te=mapminmax('apply',te,ps);
te(3,:)=[];
result=myAEPredict(stackedAEOptTheta, netconfig, te);
result=result*ps.xrange(3)+ps.xmin(3);
disp(result);
