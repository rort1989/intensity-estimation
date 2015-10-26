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
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = 0; % for bias
 
%% parameter tuning using validation data: things to vary: params range, scaled, bias, peak position: first or last
% grid search for parameters: support up to 2 varing parameters
params_A = 10.^[-5:4];
max_iter = 300; bias = 1;
if ~allframes
    for n = 1:numel(data)
        labels{n}(1,:) = src.intensity{n}(1,:);
        labels{n}(2,2) = max(src.intensity{n}(:,2));  labels{n}(2,1) = find(src.intensity{n}(:,2)==labels{n}(2,2),1,'first'); % 'last's
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
    train_data = [];
    train_label = [];
    % formalize the pairwise data
    for n = 1:N
        [~,idx] = max(labels{inst_train(n)}(:,2)); % index of apex frame
        apx = labels{inst_train(n)}(idx,1);
        [~,T] = size(data{inst_train(n)});
        pairs = zeros(T*(T+1)/2,2);
        count = 0;
        for i = apx:-1:2
            pairs(count+1:count+i-1,1) = i;
            pairs(count+1:count+i-1,2) = [i-1:-1:1]';
            count = count + i-1;
        end
        train_label = [train_label; ones(count,1)];
        count_half = count;
        if apx < T
            for i = apx:T
                pairs(count+1:count+T-i,1) = i;
                pairs(count+1:count+T-i,2) = [i+1:T]';
                count = count + T-i;
            end
        end
        pairs = pairs(1:count,:);
        train_data = [train_data data{inst_train(n)}(:,pairs(:,1)) - data{inst_train(n)}(:,pairs(:,2))];
        train_label = [train_label; 2*ones(count-count_half,1)];
    end
    % optional scaling may be performed
    if scaled
        scale_max = max(train_data,[],2);
        scale_min = min(train_data,[],2);
        temp = bsxfun(@minus, train_data, scale_min); %data_together % ATTENTION: NOT data_together
        train_data_scaled = bsxfun(@rdivide, temp, scale_max-scale_min); %train_data % try a different way of scaling, scale train data first then use the same number to scale test data
    else
        train_data_scaled = train_data;
    end
    % define initial parameter of regression model    
    if solver == 1% solver: libsvm
        svm_param = [0 0 1 1]; % L2-regularized hinge loss, cost coefficient 1
        configuration = sprintf('-s %d -t %d -g %f -c %f',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
        model = svmtrain(train_label, sparse(train_data_scaled'),configuration);
        w = model.SVs' * model.sv_coef;        b = -model.rho;
        theta = -[w(:); b];        
    elseif solver == 2        % solver: liblinear
        svm_param = [3 params_A(oter) bias]; % L2-regularized logistic regression (0,7) or square loss (2,1) hinge loss (3), cost coefficient 1, bias coefficient -1
        configuration = sprintf('-s %d -c %f -B %d',svm_param(1),svm_param(2),svm_param(3));
        model = train(train_label, sparse(train_data_scaled'),configuration);
        theta = -[model.w(:)];
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
        test_data = bsxfun(@rdivide, test_data, scale_max-scale_min);
    end
    dec_values =theta'*[test_data; ones(1,size(test_data,2))]; %
    RR = corrcoef(dec_values,test_label);  ry = RR(1,2);
    e = dec_values - test_label;
    mse = e(:)'*e(:)/length(e);
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
% retrain model using training + validation data
inst_train = union(src.idx_cv(1).train,src.idx_cv(1).validation);
inst_test = src.idx_test;
if iter == 1
    [~,opt] = max(ry_fold);
else
    [~,opt] = max(mean(ry_fold)); % or mse_fold
end
N = length(inst_train);
    train_data = [];
    train_label = [];
    % formalize the pairwise data
    for n = 1:N
        [~,idx] = max(labels{inst_train(n)}(:,2)); % index of apex frame
        apx = labels{inst_train(n)}(idx,1);
        [~,T] = size(data{inst_train(n)});
        pairs = zeros(T*(T+1)/2,2);
        count = 0;
        for i = apx:-1:2
            pairs(count+1:count+i-1,1) = i;
            pairs(count+1:count+i-1,2) = [i-1:-1:1]';
            count = count + i-1;
        end        
        train_label = [train_label; ones(count,1)];
        count_half = count;
        if apx < T
            for i = apx:T
                pairs(count+1:count+T-i,1) = i;
                pairs(count+1:count+T-i,2) = [i+1:T]';
                count = count + T-i;
            end
        end
        pairs = pairs(1:count,:);
        train_data = [train_data data{inst_train(n)}(:,pairs(:,1)) - data{inst_train(n)}(:,pairs(:,2))];
        train_label = [train_label; 2*ones(count-count_half,1)];
    end
    % optional scaling may be performed
    if scaled
        scale_max = max(train_data,[],2);
        scale_min = min(train_data,[],2);
        temp = bsxfun(@minus, train_data, scale_min); %data_together % ATTENTION: NOT data_together
        train_data_scaled = bsxfun(@rdivide, temp, scale_max-scale_min); %train_data % try a different way of scaling, scale train data first then use the same number to scale test data
    else
        train_data_scaled = train_data;
    end
    % define initial parameter of regression model    
    if solver == 1% solver: libsvm
        svm_param = [0 0 1 1]; % L2-regularized hinge loss, cost coefficient 1
        configuration = sprintf('-s %d -t %d -g %f -c %f',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
        model = svmtrain(train_label, sparse(train_data_scaled'),configuration);
        w = model.SVs' * model.sv_coef;        b = -model.rho;
        theta = -[w(:); b];        
    elseif solver == 2        % solver: liblinear
        svm_param = [3 params_A(opt) bias]; % L2-regularized logistic regression (0,7) or square loss (2,1) hinge loss (3), cost coefficient 1, bias coefficient -1
        configuration = sprintf('-s %d -c %f -B %d',svm_param(1),svm_param(2),svm_param(3));
        model = train(train_label, sparse(train_data_scaled'),configuration);
        theta = -[model.w(:)];
    end
% testing
    test_data = [];
    test_label = [];
    for n = 1:length(inst_test)
        test_data = [test_data data{inst_test(n)}];
        test_label = [test_label src.intensity{inst_test(n)}(:,2)']; % intensity
    end
    if scaled
        test_data = bsxfun(@rdivide, test_data, scale_max-scale_min);
    end
    dec_values =theta'*[test_data; ones(1,size(test_data,2))]; %
    RR = corrcoef(dec_values,test_label);  ry = RR(1,2);
    e = dec_values - test_label;
    abs_test = sum(abs(e));
    mse = e(:)'*e(:)/length(e);
    ry_test = ry;
    mse_test = mse;
 
time = toc(tt); 
display('testing completed');

%% plot concatenate seq
if solver == 3
    subplot(2,1,1)
    loglog(1:history.iter,history.s_norm,1:history.iter,history.eps_dual,'r'); title('dual feasibility')
    subplot(2,1,2)
    loglog(1:history.iter,history.r_norm,1:history.iter,history.eps_pri,'r'); title('primal feasibility')
end
% subplot(ceil(numel(inst)/5),5,iter)
% plot(test_label); hold on; 
% plot(dec_values,'r');
% % axis([0 length(intensity{inst_test(n)}) -5 9])
 
%% save results
save(sprintf('McMaster/results/exSTD_m%d_sol%d_scale%d_all%d_opt%d.mat',method,solver,scaled,allframes,option), ...
    'theta','ry_test','mse_test','abs_test','ry_fold','mse_fold','time','time_validation','solver','scaled','allframes','params_A','inst_train','inst_test','bias');
