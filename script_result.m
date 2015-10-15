%% plot rho and mse under different number of model parameters and different baselines
close all;
syn_rho = zeros(3,1);
syn_mse = zeros(3,1);
scale = zeros(1,1);
for i = 1:10
    xx=load(sprintf('SYN/results/syn%d_noisy_base_S',i),'ry','mse');
    syn_rho(1,i) = mean(xx.ry);
    syn_mse(1,i) = mean(xx.mse);
    xx=load(sprintf('SYN/results/syn%d_noisy_base2_S',i),'ry','mse','scale');
    syn_rho(2,i) = mean(xx.ry);
    syn_mse(2,i) = mean(xx.mse);
    syn_scale_mean(i) = mean(xx.scale);
    syn_scale_std(i) = std(xx.scale);
    xx=load(sprintf('SYN/results/syn%d_noisy_S',i),'ry','mse');
    syn_rho(3,i) = mean(xx.ry);
    syn_mse(3,i) = mean(xx.mse);
end
select = [1:10];
subplot(3,1,1)
plot(select+1,syn_rho(1,select),'r',select+1,syn_rho(2,select),'g',select+1,syn_rho(3,select),'b','LineWidth',2)
% xlabel('number of parameters')
title('correlation coefficient')
subplot(3,1,2)
plot(select+1,log(syn_mse(1,select)),'r',select+1,log(syn_mse(2,select)),'g',select+1,log(syn_mse(3,select)),'b','LineWidth',2)
% xlabel('number of parameters')
title('log mean square error')
subplot(3,1,3); 
errorbar(2:11,syn_scale_mean,syn_scale_std); axis([2 11 0 30])
xlabel('number of parameters')
title('scale factor between ordinal loss intensity and apex intensity')
saveas(gcf,sprintf('SYN/results/order1to5_noisy_%dframes',6),'png');
saveas(gcf,sprintf('SYN/results/order1to5_noisy_%dframes',6),'fig');
%%
set = 8;
syn_rho = zeros(3,6);
syn_mse = zeros(3,6);
for i = 1:6
    xx=load(sprintf('SYN/results/syn%d_base_S',set),'ry','mse');
    syn_rho(1,i) = mean(xx.ry);
    syn_mse(1,i) = mean(xx.mse);
    xx=load(sprintf('SYN/results/syn%d_base2_S',set),'ry','mse');
    syn_rho(2,i) = mean(xx.ry);
    syn_mse(2,i) = mean(xx.mse);
    xx=load(sprintf('SYN/results/syn%d_S_gamma%d',set,i),'ry','mse');
    syn_rho(3,i) = mean(xx.ry);
    syn_mse(3,i) = mean(xx.mse);
end
select = 1:6;
subplot(2,1,1)
plot(select+1,syn_rho(1,select),'r',select+1,syn_rho(2,select),'g',select+1,syn_rho(3,select),'b','LineWidth',2)
xlabel('magnitude order of gamma')
title('correlation coefficient')
subplot(2,1,2)
plot(select+1,log(syn_mse(1,select)),'r',select+1,log(syn_mse(2,select)),'g',select+1,log(syn_mse(3,select)),'b','LineWidth',2)
xlabel('magnitude order of gamma')
title('log mean square error')
saveas(gcf,sprintf('SYN/results/gamma1to6_%dframes',6),'png');
saveas(gcf,sprintf('SYN/results/gamma1to6_%dframes',6),'fig');