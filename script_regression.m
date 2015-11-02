%% Script for compute regression given data and intensity label for each frame
clear all;
close all;
tt = tic;
%% load data
src = load('McMaster/standard.mat','feature','intensity','idx_cv','idx_test','dfactor'); % ,'intensity'
 
% define constants
data = src.feature;
labels = cell(1,numel(data));
method = 3; % 1. both regression and ordinal loss  2. regression loss only 3. ordinal loss only
solver = 2; % with method 2 or 3, can choose whether using libsvm or liblinear to solve
scaled = 1;
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
option = 2; loss_func = [3 1]; % for svm loss function L1-loss or L2-loss
bias = 1; max_iter = 300; 

%% parameter tuning using validation data: things to vary: params range, scaled, bias, peak position: first or last
% grid search for parameters: support up to 2 varing parameters
params_A = 10.^[-5:4];
if ~allframes
    for n = 1:numel(data)
        labels{n}(1,:) = src.intensity{n}(1,:);
        labels{n}(2,2) = max(src.intensity{n}(:,2));    idx_cand = find(src.intensity{n}(:,2)==labels{n}(2,2));        
        labels{n}(2,1) = idx_cand(max(1,ceil(length(idx_cand)/2)));
        labels{n}(3,:) = src.intensity{n}(end,:);
    end
else
    labels = src.intensity;
end
 
for oter = 1:numel(params_A)
for iter = 1:length(src.idx_cv)
    inst_train = src.idx_cv(iter).train;
    inst_test = src.idx_cv(iter).validation;
    % ordinal regression formulation: ordinal loss only with regularization
    N = length(inst_train);
    train_data = []; % for liblinear or libsvm
    train_data_ori = [];
    train_data_aug = [];
    train_label = [];
    ordinal_label = [];
    qid = [];
    % formalize the pairwise data
    count_qid = 0;
    for n = 1:N
        train_data_ori = [train_data_ori data{inst_train(n)}];
    end
    if scaled
        scale_max = max(train_data_ori,[],2);
        scale_min = min(train_data_ori,[],2);        
    end
    for n = 1:N
        data_buffer = data{inst_train(n)};
        if scaled
            temp = bsxfun(@minus, data_buffer, scale_min); %data_together % ATTENTION: NOT data_together
            data_buffer = bsxfun(@rdivide, temp, scale_max-scale_min);
        end
        peak = max(labels{inst_train(n)}(:,2)); % index of apex frame
        idx = find(labels{inst_train(n)}(:,2)==peak);
        apx = labels{inst_train(n)}(idx(max(1,ceil(length(idx)/2))),1);
        [~,T] = size(data_buffer);        
        pairs = zeros(T*(T+1)/2,2);
        count = 0;        
        for i = apx:-1:2
            pairs(count+1:count+i-1,1) = i;
            pairs(count+1:count+i-1,2) = [i-1:-1:1]';
            count = count + i-1;
        end
        train_label = [train_label; ones(count,1)];  
        qid = [qid; (count_qid+1)*ones(apx,1)]; % change here in case does not support one instance qid case
        ordinal_label = [ordinal_label; (1:apx)']; 
        train_data_aug = [train_data_aug data_buffer(:,1:apx)];
        count_qid = count_qid + 1;
        count_half = count;
        if apx < T
            for i = apx:T
                pairs(count+1:count+T-i,1) = i;
                pairs(count+1:count+T-i,2) = [i+1:T]';
                count = count + T-i;
            end
            qid = [qid; (count_qid+1)*ones(T-apx+1,1)]; % change here in case does not support one instance qid case
            ordinal_label = [ordinal_label; (T-apx+1:-1:1)']; 
            train_data_aug = [train_data_aug data_buffer(:,apx:T)];
            count_qid = count_qid + 1;
        end
        pairs = pairs(1:count,:);
        train_data = [train_data data_buffer(:,pairs(:,1)) - data_buffer(:,pairs(:,2))];
        train_label = [train_label; 2*ones(count-count_half,1)];
    end    
    % if size(train_data_aug,2)~=length(qid) || size(train_data_aug,2)~=length(ordinal_label) || max(qid) ~= count_qid

    % define initial parameter of regression model    
    if solver == 1% solver: libsvm
        svm_param = [0 0 1 1]; % L2-regularized hinge loss, cost coefficient 1
        configuration = sprintf('-s %d -t %d -g %f -c %f',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
        model = svmtrain(train_label, sparse(train_data'),configuration);
        w = model.SVs' * model.sv_coef;        b = -model.rho;
        theta = [w(:); b];        
    elseif solver == 2        % solver: liblinear
        svm_param = [loss_func(option) params_A(oter) bias]; % L2-regularized logistic regression (0,7) or square loss (2,1) hinge loss (3), cost coefficient 1, bias coefficient -1
        configuration = sprintf('-s %d -c %f -B %d',svm_param(1),svm_param(2),svm_param(3));
        model = train(train_label, sparse(train_data'),configuration);
        theta = [model.w(:)];
    elseif solver == 3   % solver: rank-svm
        train_data_aug = [ordinal_label'; qid'; train_data_aug];    
    end
    
    %% validation: compute the prediction intensity given testing frame and learned model
    % alternative: concatenate all testing frames
    test_data = [];
    test_label = [];
    for n = 1:length(inst_test)
        test_data = [test_data data{inst_test(n)}];
        test_label = [test_label src.intensity{inst_test(n)}(:,2)']; % intensity
    end
    if scaled
        temp = bsxfun(@minus, test_data, scale_min);
        test_data = bsxfun(@rdivide, temp, scale_max-scale_min);
    end
    if solver < 3
        dec_values =theta'*[test_data; ones(1,size(test_data,2))]; %
    else % solver: rank-svm
        test_data_aug = [1:size(test_data,2); ones(1,size(test_data,2)); test_data];
        params = sprintf('-c %f -e 0.1',params_A(oter));
        dec_values = svmrank(train_data_aug',test_data_aug',params);  dec_values = dec_values';
    end
    RR = corrcoef(dec_values,test_label);  ry = RR(1,2);
    e = dec_values - test_label;
    mse = e(:)'*e(:)/length(e);
    abs_fold(iter,oter) = sum(abs(e))/length(e);
    ry_fold(iter,oter) = ry;
    mse_fold(iter,oter) = mse;
    display(sprintf('validation iteration %d completed',iter));
 
end
display(sprintf('--grid %d completed',oter))
end  % cross-validation
time_validation = toc(tt);
tt = tic;
 
%% re-train model and test on testing data 
% identify the best model parameter
if iter == 1
    [~,opt] = max(ry_fold);
else
    [~,opt] = max(mean(ry_fold)); % or mse_fold
end
params_A(opt)
% retrain model using training + validation data
dec_values_test = [];
labels_test = [];
for iter = 1:length(src.idx_test)    
    inst_train = src.idx_test(iter).train;
    inst_test = src.idx_test(iter).validation; %train
    N = length(inst_train);
    train_data = []; % for liblinear or libsvm
    train_data_ori = [];
    train_data_aug = [];
    train_label = [];
    ordinal_label = [];
    qid = [];
    % formalize the pairwise data
    count_qid = 0;
    for n = 1:N
        train_data_ori = [train_data_ori data{inst_train(n)}];
    end
    if scaled
        scale_max = max(train_data_ori,[],2);
        scale_min = min(train_data_ori,[],2);        
    end
    for n = 1:N
        data_buffer = data{inst_train(n)};
        if scaled
            temp = bsxfun(@minus, data_buffer, scale_min); %data_together % ATTENTION: NOT data_together
            data_buffer = bsxfun(@rdivide, temp, scale_max-scale_min);
        end
        peak = max(labels{inst_train(n)}(:,2)); % index of apex frame
        idx = find(labels{inst_train(n)}(:,2)==peak);
        apx = labels{inst_train(n)}(idx(max(1,ceil(length(idx)/2))),1);
        [~,T] = size(data_buffer);        
        pairs = zeros(T*(T+1)/2,2);
        count = 0;        
        for i = apx:-1:2
            pairs(count+1:count+i-1,1) = i;
            pairs(count+1:count+i-1,2) = [i-1:-1:1]';
            count = count + i-1;
        end
        train_label = [train_label; ones(count,1)];  
        qid = [qid; (count_qid+1)*ones(apx,1)]; % change here in case does not support one instance qid case
        ordinal_label = [ordinal_label; (1:apx)']; 
        train_data_aug = [train_data_aug data_buffer(:,1:apx)];
        count_qid = count_qid + 1;
        count_half = count;
        if apx < T
            for i = apx:T
                pairs(count+1:count+T-i,1) = i;
                pairs(count+1:count+T-i,2) = [i+1:T]';
                count = count + T-i;
            end
            qid = [qid; (count_qid+1)*ones(T-apx+1,1)]; % change here in case does not support one instance qid case
            ordinal_label = [ordinal_label; (T-apx+1:-1:1)']; 
            train_data_aug = [train_data_aug data_buffer(:,apx:T)];
            count_qid = count_qid + 1;
        end
        pairs = pairs(1:count,:);
        train_data = [train_data data_buffer(:,pairs(:,1)) - data_buffer(:,pairs(:,2))];
        train_label = [train_label; 2*ones(count-count_half,1)];
    end
    
    % define initial parameter of regression model    
    if solver == 1% solver: libsvm
        svm_param = [0 0 1 1]; % L2-regularized hinge loss, cost coefficient 1
        configuration = sprintf('-s %d -t %d -g %f -c %f',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
        model = svmtrain(train_label, sparse(train_data'),configuration);
        w = model.SVs' * model.sv_coef;        b = -model.rho;
        theta = [w(:); b];        
    elseif solver == 2        % solver: liblinear
        svm_param = [loss_func(option) params_A(opt) bias]; % L2-regularized logistic regression (0,7) or square loss (2,1) hinge loss (3), cost coefficient 1, bias coefficient -1
        configuration = sprintf('-s %d -c %f -B %d',svm_param(1),svm_param(2),svm_param(3));
        model = train(train_label, sparse(train_data'),configuration);
        theta = [model.w(:)];    
    elseif solver == 3   % solver: rank-svm
        train_data_aug = [ordinal_label'; qid'; train_data_aug];
    end
    % testing
    test_data = [];
    test_label = [];
    for n = 1:length(inst_test)
        test_data = [test_data data{inst_test(n)}];
        test_label = [test_label src.intensity{inst_test(n)}(:,2)']; % intensity
    end
    if scaled
        temp = bsxfun(@minus, test_data, scale_min);
        test_data = bsxfun(@rdivide, temp, scale_max-scale_min);
    end
    if solver < 3        
        dec_values =theta'*[test_data; ones(1,size(test_data,2))];         
    else % solver: rank-svm
        test_data_aug = [1:size(test_data,2); ones(1,size(test_data,2)); test_data];
        params = sprintf('-c %f -e 0.1',params_A(opt));
        dec_values = svmrank(train_data_aug',test_data_aug',params); dec_values = dec_values';
    end
    RR = corrcoef(dec_values,test_label);  ry_test(iter) = RR(1,2);
    e = dec_values - test_label;
    dec_values_test = [dec_values_test dec_values];
    labels_test = [labels_test test_label];
    abs_test(iter) = sum(abs(e))/length(e);
    mse = e(:)'*e(:)/length(e);
    mse_test(iter) = mse;
    display(sprintf('testing iteration %d completed',iter));
    subplot(ceil(numel(src.idx_test)/5),5,iter)
    plot(test_label); hold on;
    plot(dec_values,'r');
    axis([0 length(test_label) -5 9])
end
time = toc(tt); 
display('testing completed');

%% plot concatenate seq
if solver == 3
%     figure
%     subplot(2,1,1)
%     loglog(1:history.iter,history.s_norm,1:history.iter,history.eps_dual,'r'); title('dual feasibility')
%     subplot(2,1,2)
%     loglog(1:history.iter,history.r_norm,1:history.iter,history.eps_pri,'r'); title('primal feasibility')
end
 
%% save results
mean(ry_test)
mean(mse_test)
mean(abs_test)
save(sprintf('McMaster/results/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d.mat',method,solver,scaled,allframes,option,bias), ...
    'ry_test','mse_test','abs_test','dec_values_test','labels_test','ry_fold','mse_fold','abs_fold','time','time_validation','solver','scaled','allframes','params_A','inst_train','inst_test','bias');
if solver < 3
   save(sprintf('McMaster/results/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d.mat',method,solver,scaled,allframes,option,bias),'theta','svm_param','-append');
end