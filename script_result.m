% %% scripts to generate plot, analyze results
% % selected sequence for each subject
% inst{1} = [6]; % ex1
% inst{2} = [5]; % ex1
% inst{3} = [1 4]; % ex1
% inst{4} = [2 6]; % ex1
% inst{5} = [3 5 7]; % ex1 inst 1,2
% inst{6} = [1 8]; % ex1
% inst{7} = [];
% inst{8} = [2 3]; % ex1
% inst{9} = [8]; % ex1
% inst{10} = [2:7]; % ex1 inst 2,5
% inst{11} = [2 5];
% inst{12} = [3]; % ex1
% inst{13} = [5]; % ex1
% inst{14} = [1 2 5]; % ex1 inst 1
% inst{15} = [];
% inst{16} = [1 5 7 9 11 13 15]; % ex1 inst 1,9
% inst{17} = [1];
% inst{18} = [1 4]; % ex1 - inst4
% inst{19} = [3 4 10];
% inst{20} = [2 6];
% inst{21} = [1 3 5]; % ex1 - inst3
% inst{22} = [1];
% inst{23} = [];
% inst{24} = [2 6];
% inst{25} = [1 5]; % ex1
% 
% % plot the pain intensity changes
% close all;
% src = load('McMaster/McMaster.mat');
% for s = 1:numel(src.subject)
%     PSPI = src.PSPI{s};
%     figure
%     for n = 1:numel(inst{s})%numel(PSPI)
%         seq = PSPI{inst{s}(n)}; %seq = PSPI{n};
%         subplot(1,numel(inst{s}),n);%subplot(2,ceil(numel(PSPI)/2),n);
%         plot(seq);
%         title(num2str(n));
%     end
% end
% 
% %%
% inst_select = [1:4 10 12:21];
% for i = 1:length(inst_select)%[1:4 10:12 14:16]%
%     subplot(length(inst_select)/3,3,i);
%     plot(intensity{inst_select(i)});
%     axis([1 length(intensity{inst_select(i)}) 0 9])
% end
% 
% % subplot(2,1,1)
% % plot(intensity); title('intensity level of selected AU')
% % subplot(2,1,2)
% % plot(dec_values); title('intensity from learned regression model')

%% script to generate results
% CK+, BU-4DEF for each type of expression, compute its 
clear all;
method = 1;
solver = 3;
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = 1;
bias = 1;
express = 0;
res = load(sprintf('CK+/results/PCA_LBP/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d_exp%d.mat',method,solver,scaled,allframes,option,bias,express));
src = load('CK+/PCA_LBP/standard_all.mat');
ry = zeros(1,7);
mae = zeros(1,7);
mse = zeros(1,7);
for e = 1:7
    idx = find(src.label_exp==e);
    dec_values = res.dec_values_test(idx);
    labels = res.labels_test(idx);
    RR = corrcoef(dec_values,labels);  ry(e) = RR(1,2);
    err = dec_values - labels;
    mse(e) = err(:)'*err(:)/length(err);
    mae(e) = sum(abs(err))/length(err);
end
save(sprintf('CK+/results/PCA_LBP/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d_exp%d.mat',method,solver,scaled,allframes,option,bias,express),'ry','mse','mae','-append');

%% average results of leave-one-subject-out test
clear all;
method = 3;
solver = 3;
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = 1;
bias = 1;
num_exp = 0;
results_mat = zeros(6,num_exp+1);
for express = 1%:num_exp %CK+
    src = load(sprintf('McMaster/results/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d.mat',method,solver,scaled,allframes,option,bias));
    % average within each subjects
    results_mat(1,express) = mean(src.ry_test);
    results_mat(2,express) = mean(src.mse_test);
    results_mat(3,express) = mean(src.abs_test);
    % average across different subjects
    e = src.dec_values_test - src.labels_test;    
    results_mat(4,express) = mean((src.dec_values_test-mean(src.dec_values_test)).*(src.labels_test-mean(src.labels_test)))/std(src.dec_values_test,1)/std(src.labels_test,1);
    results_mat(5,express) = e(:)'*e(:)/length(e);
    results_mat(6,express) = sum(abs(e))/length(e);
end
% results_mat(:,num_exp+1) = mean(results_mat(:,1:num_exp),2);
% save(sprintf('McMaster/results/m%d_sol%d_scale%d_all%d_opt%d_bias%d.mat',method,solver,scaled,allframes,option,bias),'results_mat');

%% BACKUP CODE
% limited to McMaster
clear all; close all;
method = [1 1 2 3];
solver = [3 3 2 3];
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = [1 2 1 1];
bias = 1;
range = 1317:1350;
cc = hsv(length(method)+1);
for m = [1 2 4]%1:length(method)
    src = load(sprintf('McMaster/results/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d.mat',method(m),solver(m),scaled,allframes,option(m),bias));
    if m == 1
        plot(src.dec_values_test(range),'r','LineWidth',2); % 'color',cc(m,:)
    elseif m == 2
        plot(src.dec_values_test(range),'b--','LineWidth',2);
%     elseif m == 3
%         plot(src.dec_values_test(range),'g-.','LineWidth',2);
    else
        plot(src.dec_values_test(range),'c:','LineWidth',2);
    end        
    hold on;
end
% load('McMaster/results/rankboost.mat');
% plot(record(2,range),'color',[0.75 0.75 0.75],'LineWidth',2);%'color',cc(m+1,:),
% hold on;
load('McMaster/results/record.mat'); % RVR-P
plot(record(range,2),'m--','LineWidth',2);%'color',cc(m+1,:),
hold on;
plot(src.labels_test(range),'k','LineWidth',2);
grid on;
hold off;
axis([1 length(range) -4 7])
xlabel('frame')
ylabel('intensity')
title('weakly supervised or unsupervised settings')

%% select and plot a part of testing labels and predicted values of different method
% limited to McMaster
% RED: OSVR-L1, BLUE: OSVR-L2, GREEN: SVR, CYN: RVR, MAGENTA: Rankboost, GREY: OR,  BLACK: TRUTH
clear all; close all;
method = [1 1 2];
solver = [3 3 2];
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = [1 2 1];
bias = 1;
range = 1317:1350;
for m = 1:length(method)
    src = load(sprintf('McMaster/results/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d.mat',method(m),solver(m),scaled,allframes,option(m),bias));
    if m == 1
        plot(src.dec_values_test(range),'r','LineWidth',2);
    elseif m == 2
        plot(src.dec_values_test(range),'b--','LineWidth',2);
    else
        plot(src.dec_values_test(range),'g-.','LineWidth',2);
    end        
    hold on;
end
load('McMaster/results/record_keyframe.mat'); % RVR-P
plot(a(:,2),'c:','LineWidth',2);
hold on;
plot(src.labels_test(range),'k','LineWidth',2);
grid on;
hold off;
axis([1 length(range) -2 6])
xlabel('frame')
ylabel('intensity')
title('Weakly supervised setting')

% supervised case
figure
method = [1 1 2];
solver = [3 3 2];
allframes = 1; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = [1 2 1];
bias = 1;
range = 1317:1350;
cc = hsv(length(method)+1);
for m = 1:length(method)
    src = load(sprintf('McMaster/results/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d.mat',method(m),solver(m),scaled,allframes,option(m),bias));
    if m == 1
        plot(src.dec_values_test(range),'r','LineWidth',2);
    elseif m == 2
        plot(src.dec_values_test(range),'b--','LineWidth',2);
    elseif m == 3
        plot(src.dec_values_test(range),'g-.','LineWidth',2);
    end        
    hold on;
end
load('McMaster/results/record.mat');
plot(record(range,2),'c:','LineWidth',2);
hold on;
plot(src.labels_test(range),'k','LineWidth',2);
grid on;
hold off;
axis([1 length(range) -2 6])
xlabel('frame')
ylabel('intensity')
title('Fully supervised setting')

% unsupervised
figure
method = [2];
solver = [2];
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = 1;
bias = 1;
range = 1317:1350;
m = 1;
    src = load(sprintf('McMaster/results/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d.mat',method(m),solver(m),scaled,allframes,option(m),bias));
        plot(src.dec_values_test(range),'color',[0.75 0.75 0.75],'LineWidth',2); % 'color',cc(m,:)
        hold on;
load('McMaster/results/rankboost.mat');
plot(record(2,range),'m--','LineWidth',2);%'color',cc(m+1,:),
hold on;
plot(src.labels_test(range),'k','LineWidth',2);
grid on;
hold off;
axis([1 length(range) -2 6])
xlabel('frame')
ylabel('intensity')
title('Unsupervised setting')