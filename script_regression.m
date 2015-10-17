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
method = 3; % 1. w/o reg.  2. SVR 3. rank-SVM
solver = 2; % with method 2 or 3, can choose whether using libsvm or liblinear to solve
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 0;

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
    T = numel(intensity{n});
    if ~allframes
        [labels{n}(1,2),labels{n}(1,1)] = min(intensity{n});
        [labels{n}(2,2),labels{n}(2,1)] = max(intensity{n});
        labels{n}(3,1) = numel(intensity{n});
        labels{n}(3,2) = intensity{n}(end);
        if labels{n}(2,1) < 2
            nconstraint(n) = nconstraint(n) + nchoosek(T,2);
        elseif labels{n}(2,1) > T-1
            nconstraint(n) = nconstraint(n) + nchoosek(labels{n}(2,1),2);
        else
            nconstraint(n) = nconstraint(n) + nchoosek(labels{n}(2,1),2) + nchoosek(T-labels{n}(2,1)+1,2);
        end
    else
        labels{n} = [1:numel(intensity{n})]';
        labels{n} = [labels{n} intensity{n}'];
    end
end
% select a subset of features
fdim = size(data{1},1); % dimension of input features

%% solution
if method == 1
% Ordinal SVR:  hinge loss on both regressin and ordinal
if solver == 1
    % define initial parameter of regression model
    rng default;
    theta0 = 0.1*randn(fdim+1,1);%
    gamma = [1 10];
    % train regression model
    % Solving minimization problem using Matlab optimization toolbox
    options = optimset('GradObj','on','LargeScale','off');
%     [f0,g0] = regressor(theta0,data,labels,gamma);
%     numgrad = computeNumericalGradient(@(theta) regressor(theta,data,labels,gamma), theta0);
%     err = norm(g0-numgrad);
    [theta,f,eflag,output,g] = fminunc(@(theta) regressor(theta,data(inst_train),labels(inst_train),gamma), theta0, options); % _base2
elseif solver == 2
    gamma = [1 1]; % note that two gammas, one for each loss term
    epsilon = 0.1;
    [w, b, alpha] = osvrtrain(labels(inst_train), data(inst_train), epsilon, gamma);
    theta = [w(:);b];
end

elseif method == 2
% SVR formulation: regression loss only with regularization
N = length(inst_train);
train_data = [];
train_label = [];
for n = 1:N
    train_data = [train_data; data{inst_train(n)}(:,labels{inst_train(n)}(:,1))'];
    train_label = [train_label; labels{inst_train(n)}(:,2)];
end
% additional scaling may be performed
if scaled
    temp = bsxfun(@minus, train_data, min(train_data)); %data_together % ATTENTION: NOT data_together
    data_together_scaled = bsxfun(@rdivide, temp, max(train_data)-min(train_data)); %train_data % try a different way of scaling, scale train data first then use the same number to scale test data
    train_data_scaled = data_together_scaled(1:size(train_data,1),:);
else
    train_data_scaled = train_data;
end
% define initial parameter of regression model
if solver == 1
    % solver: libsvm
    svm_param = [3 0 1 0.1]; % L2-regularized hinge-loss, linear kernel, gamma coefficient for kernel, tolerence 0.1
    configuration = sprintf('-s %d -t %d -g %f -c %f',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
    model = svmtrain(train_label, sparse(train_data_scaled),configuration);
    % get parameter w,b from model
    w = model.SVs' * model.sv_coef;
    b = -model.rho;
    theta = [w(:); b];
    % evaluate on training data first
    [predict_label, ~, dec_values_train] = svmpredict(train_label, sparse(train_data_scaled), model);
elseif solver == 2
    % solver: liblinear
    svm_param = [11 1 0.1 1]; % L2-regularized L2-loss(11) or L1-loss(13), cost coefficient 1, tolerance 0.1, bias coefficient 1
    configuration = sprintf('-s %d -c %f -p %f -B %d',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
    model = train(train_label, sparse(train_data_scaled),configuration);
    theta = model.w(:);
end

elseif method == 3 % % % %%%%%%%%%%%%%%%%%%%%% FURTHER DEBUG THIS PART
% ordinal regression formulation: ordinal loss only with regularization
N = length(inst_train);
train_data = [];
train_label = [];
% formalize the pairwise data
for n = 1:N
    [~,idx] = max(labels{inst_train(n)}(:,2)); % index of apex frame
    apx = labels{inst_train(n)}(idx,1);
    if apx > 1
        pairs_fhalf = nchoosek(1:apx,2);
        for p = 1:size(pairs_fhalf)
            train_data = [train_data; data{inst_train(n)}(:,pairs_fhalf(p,1))'-data{inst_train(n)}(:,pairs_fhalf(p,2))'];
        end
        train_label = [train_label; ones(p,1)];
    end
    T = size(data{inst_train(n)},2);
    if apx < T
        pairs_lhalf = nchoosek(apx:T,2);  % allocate space for pairs first when scale up
        for p = 1:size(pairs_lhalf,1)
            train_data = [train_data; data{inst_train(n)}(:,pairs_lhalf(p,1))'-data{inst_train(n)}(:,pairs_lhalf(p,2))'];
        end
        train_label = [train_label; 2*ones(p,1)];
    end
end
if scaled
    temp = bsxfun(@minus, train_data, min(train_data)); %data_together % ATTENTION: NOT data_together
    data_together_scaled = bsxfun(@rdivide, temp, max(train_data)-min(train_data)); %train_data % try a different way of scaling, scale train data first then use the same number to scale test data
    train_data_scaled = data_together_scaled(1:size(train_data,1),:);
else
    train_data_scaled = train_data;
end
% define initial parameter of regression model
% solver: libsvm
if solver == 1
    svm_param = [0 0 1 1]; % L2-regularized hinge loss, cost coefficient 1
    configuration = sprintf('-s %d -t %d -g %f -c %f',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
    model = svmtrain(train_label, sparse(train_data_scaled),configuration);
    % get parameter w,b from model
    w = model.SVs' * model.sv_coef;
    b = -model.rho;
    theta = -[w(:); b];
    % evaluate on training data first
    [predict_label, ~, dec_values_train] = svmpredict(train_label, sparse(train_data_scaled), model);
elseif solver == 2
    % solver: liblinear
    svm_param = [0 1 -1]; % L2-regularized logistic regression, cost coefficient 1, bias coefficient -1
    configuration = sprintf('-s %d -c %f -B %d',svm_param(1),svm_param(2),svm_param(3));
    model = train(train_label, sparse(train_data_scaled),configuration);
    theta = -[model.w(:); 0];
end
end

%% test: compute the prediction intensity given testing frame and learned model
dec_values = cell(1,length(inst_test));
ry = zeros(1,length(inst_test));
mse = zeros(1,length(inst_test));
scale = zeros(1,length(inst_test));
for n = 1:length(inst_test)
    dec_values{n} =theta'*[data{inst_test(n)}; ones(1,size(data{inst_test(n)},2))]; %
    RR = corrcoef(dec_values{n},intensity{inst_test(n)});  ry(n) = RR(1,2);
    e = dec_values{n} - intensity{inst_test(n)};
    mse(n) = e(:)'*e(:)/length(e);
    [value,apex] = max(labels{inst_test(n)}(:,2));
    scale(n) = dec_values{n}(apex)/value;
end
ry_fold(iter) = mean(ry);
mse_fold(iter) = mean(mse);
scale_fold(iter) = std(scale);
display(sprintf('iteration %d completed',iter));
%%
for n = 1:length(inst_test)
	subplot(length(inst_select)/3,3,idx_cv(iter).validation(n))%n
    plot(intensity{inst_test(n)}); hold on; 
    plot(dec_values{n},'r');
    axis([0 length(intensity{inst_test(n)}) -5 9])
end

end % cross-validation
%% save results
% save('McMaster/results/ex1_fea1_base2_cv5.mat','theta0','theta','f','eflag','output','g','inst_select','inst_train','inst_test','ry','mse','scale','gamma','dfactor');
mean(ry_fold)
mean(mse_fold)
%% plot intensity
% close all;
% for n = 1:length(inst_test)
%     figure;
%     plot(intensity{inst_test(n)},'r'); hold on; 
%     plot(dec_values{n});
%     %axis([0 length(idx) -1 6])
% end
time = toc(tt);

