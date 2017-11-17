data=xlsread('DEHP.xlsx');
maxt2=1000;
index=~isnan(data(:,6));
data=data(index,:);
[data, ps]=mapminmax(data', 0, 1);
data=data';
data(:,5)=data(:,5)*ps.xrange(5)+ps.xmin(5);
data(:,5)=data(:,5)/maxt2;
save DEHP data ps maxt2;