%% Script for compute regression given data and intensity label for each frame
clear all;
close all;
tt = tic;
% load data
src = load('McMaster/McMaster.mat');
% first experiment 
inst{1} = [6]; inst{2} = [5]; inst{3} = [1 4]; inst{4} = [2 6]; inst{5} = [3 5]; inst{6} = [1]; inst{7} = [];
inst{8} = [2]; inst{9} = [8]; inst{10} = [2 5]; inst{11} = []; inst{12} = [3]; inst{13} = [5]; inst{14} = [1]; 
inst{15} = []; inst{16} = [1 9]; inst{17} = []; inst{18} = [4]; inst{19} = []; inst{20} = []; inst{21} = [3]; 
inst{22} = []; inst{23} = []; inst{24} = []; inst{25} = [1];

inst_select = [1:4 10 12:21];
idx_cv = cv_idx(length(inst_select),5);

for iter = 1:length(idx_cv)
data = cell(1); % features;
id_sub = 0; % id of each sub: can use to find number of seq per sub
intensity = cell(1); % src.PSPI;
labels = cell(1); % begining, apex, end
count_inst = 0;
for s = 1:numel(inst)
    if ~isempty(inst{s})        
        for n = 1:numel(inst{s})
            count_inst = count_inst + 1;
            data{count_inst} = src.PCA_LBP_features{s}{inst{s}(n)}'; % features
            intensity{count_inst} = src.PSPI{s}{inst{s}(n)}'; % pain intensity: a scalar
            [labels{count_inst}(1,2),labels{count_inst}(1,1)] = min(intensity{count_inst});
            [labels{count_inst}(2,2),labels{count_inst}(2,1)] = max(intensity{count_inst});
            labels{count_inst}(3,1) = numel(intensity{count_inst},1);
            labels{count_inst}(3,2) = intensity{count_inst}(end);
            id_sub(count_inst) = s;
        end        
    end
end
inst_train = inst_select(idx_cv(iter).train); %1:count_inst-5; %
inst_test = inst_select(idx_cv(iter).validation); %1+count_inst-5:count_inst;

%% feature extraction / dimension reduction / downsampling
% downsample: if the same intensity level stays for up to dfactor frames,
% downsample it to one frame
dfactor = 10;
nconstraint = zeros(count_inst,1);
for n = 1:count_inst
    T = numel(intensity{n});
    slope = diff([-1 intensity{n}]);
    changepoint = find(slope ~= 0); changepoint = [changepoint T];
    idx_select = [];
    for i = 1:length(changepoint)-1
        if changepoint(i+1) - changepoint(i) >= dfactor
            idx_select = [idx_select changepoint(i):dfactor:changepoint(i+1)-1];
        else
            idx_select = [idx_select changepoint(i):changepoint(i+1)-1];
        end
    end
    data{n} = data{n}(:,idx_select);
    intensity{n} = intensity{n}(idx_select);
    [labels{n}(1,2),labels{n}(1,1)] = min(intensity{n});
    [labels{n}(2,2),labels{n}(2,1)] = max(intensity{n});
    labels{n}(3,1) = numel(intensity{n});
    labels{n}(3,2) = intensity{n}(end);
    T = numel(intensity{n});
    if labels{n}(2,1) < 2
        nconstraint(n) = nconstraint(n) + nchoosek(T,2);
    elseif labels{n}(2,1) > T-1
        nconstraint(n) = nconstraint(n) + nchoosek(labels{n}(2,1),2);
    else
        nconstraint(n) = nconstraint(n) + nchoosek(labels{n}(2,1),2) + nchoosek(T-labels{n}(2,1)+1,2);
    end
end
% select a subset of features
fdim = size(data{1},1); % dimension of input features
% % SVR formulation
% N = numel(data);
% train_data = [];
% train_label = [];
% for n = 1:N
%     nframe = size(labels{n},1);
%     for m = 1:nframe
%         train_data = [train_data; data{n}(:,labels{n}(m,1))'];
%         train_label = [train_label; labels{n}(m,2)];
%     end
% end
% % additional scaling may be performed
% train_data_scaled = train_data;
% 
% %% define initial parameter of regression model
% svm_param = [3 0 1 1];
% configuration = sprintf('-s %d -t %d -g %f -c %f',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
% model = svmtrain(train_label, train_data_scaled,configuration);
% % get parameter w,b from model
% w = model.SVs' * model.sv_coef;
% b = -model.rho;
% theta = [w(:); b];
% % evaluate on training data first
% [predict_label, ~, dec_values_train] = svmpredict(train_label, sparse(train_data_scaled), model);

%% define initial parameter of regression model
rng default;
theta0 = 0.1*randn(fdim+1,1); 
gamma = 1;
% train regression model
% Solving minimization problem using Matlab optimization toolbox
options = optimset('GradObj','on','LargeScale','off');
% [f0,g0] = regressor(theta0,data,labels);
% numgrad = computeNumericalGradient(@(theta) regressor(theta,data,labels), theta0);
% err = norm(g0-numgrad);
[theta,f,eflag,output,g] = fminunc(@(theta) regressor(theta,data(inst_train),labels(inst_train),gamma), theta0, options);

%% test: compute the AU intensity given testing frame and learned model
dec_values = cell(1,length(inst_test));
ry = zeros(1,length(inst_test));
mse = zeros(1,length(inst_test));
scale = zeros(1,length(inst_test));
for n = 1:length(inst_test)
    dec_values{n} =theta'*[data{inst_test(n)}; ones(1,size(data{inst_test(n)},2))]; %
    RR = corrcoef(dec_values{n},intensity{inst_test(n)});  ry(n) = RR(1,2);
    e = dec_values{n} - intensity{inst_test(n)};
    mse(n) = e(:)'*e(:)/length(e);
    apex = labels{inst_test(n)}(2,1);
    scale(n) = dec_values{n}(apex)/labels{inst_test(n)}(2,2);
end
ry_fold(iter) = mean(ry)
mse_fold(iter) = mean(mse)
scale_fold(iter) = std(scale)
diaplay(sprintf('iteration %d completed',iter));
%% save results
end % cross-validation
save('McMaster/results/ex1_fea1_cv5.mat','theta0','theta','f','eflag','output','g','inst','dfactor','ry','labels');

%% plot intensity
close all;
for n = 1:length(inst_test)
    figure;
    plot(intensity{inst_test(n)},'r'); hold on; 
    plot(dec_values{n});
    %axis([0 length(idx) -1 6])
end
time = toc(tt);

