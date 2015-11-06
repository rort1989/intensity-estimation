%% scripts to generate plot, analyze results
% selected sequence for each subject
inst{1} = [6]; % ex1
inst{2} = [5]; % ex1
inst{3} = [1 4]; % ex1
inst{4} = [2 6]; % ex1
inst{5} = [3 5 7]; % ex1 inst 1,2
inst{6} = [1 8]; % ex1
inst{7} = [];
inst{8} = [2 3]; % ex1
inst{9} = [8]; % ex1
inst{10} = [2:7]; % ex1 inst 2,5
inst{11} = [2 5];
inst{12} = [3]; % ex1
inst{13} = [5]; % ex1
inst{14} = [1 2 5]; % ex1 inst 1
inst{15} = [];
inst{16} = [1 5 7 9 11 13 15]; % ex1 inst 1,9
inst{17} = [1];
inst{18} = [1 4]; % ex1 - inst4
inst{19} = [3 4 10];
inst{20} = [2 6];
inst{21} = [1 3 5]; % ex1 - inst3
inst{22} = [1];
inst{23} = [];
inst{24} = [2 6];
inst{25} = [1 5]; % ex1

% plot the pain intensity changes
close all;
src = load('McMaster/McMaster.mat');
for s = 1:numel(src.subject)
    PSPI = src.PSPI{s};
    figure
    for n = 1:numel(inst{s})%numel(PSPI)
        seq = PSPI{inst{s}(n)}; %seq = PSPI{n};
        subplot(1,numel(inst{s}),n);%subplot(2,ceil(numel(PSPI)/2),n);
        plot(seq);
        title(num2str(n));
    end
end

%%
inst_select = 1:21;% [1:4 10 12:21];
for i = 1:length(inst_select)%[1:4 10:12 14:16]%
    subplot(length(inst_select)/3,3,i);
    plot(intensity{inst_select(i)});
    axis([1 length(intensity{inst_select(i)}) 0 9])
end

% subplot(2,1,1)
% plot(intensity); title('intensity level of selected AU')
% subplot(2,1,2)
% plot(dec_values); title('intensity from learned regression model')

%% average results of leave-one-subject-out test
clear all;
method = 2;
solver = 2;
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = 1;
bias = 1;
num_exp = 6;
results_mat = zeros(6,num_exp+1);
for express = 1:num_exp %CK+
    src = load(sprintf('BU4DEF/results/PCA_ALL/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d_exp%d.mat',method,solver,scaled,allframes,option,bias,express));
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
results_mat(:,num_exp+1) = mean(results_mat(:,1:num_exp),2);
% save(sprintf('BU4DEF/results/m%d_sol%d_scale%d_all%d_opt%d_bias%d_exp%d.mat',method,solver,scaled,allframes,option,bias,express),'results_mat');

%% average results of leave-one-subject-out test
% clear all;
method = 3;
solver = 3;
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = 1;
bias = 1;
num_exp = 0;
results_mat = zeros(6,num_exp+1);
for express = 1%:num_exp %
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
% save(sprintf('BU4DEF/results/m%d_sol%d_scale%d_all%d_opt%d_bias%d_exp%d.mat',method,solver,scaled,allframes,option,bias,express),'results_mat');