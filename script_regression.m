%% Script for compute regression given data and intensity label for each frame
clear all;
close all;
tt = tic;
%% load data
src = load('McMaster/standard.mat','feature','intensity','idx_cv','idx_test','dfactor'); % ,'intensity'

% define constants
data = src.feature;
labels = cell(1,numel(data));
method = 2; % 1. both regression and ordinal loss  2. regression loss only 3. ordinal loss only
solver = 2; % with method 2 or 3, can choose whether using libsvm or liblinear to solve
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = 0; 
options = optimset('GradObj','on','LargeScale','off','MaxIter',1000); theta0 = zeros(size(data{1},1)+1,1);

%% parameter tuning using validation data: things to vary: params range, scaled, bias, peak position: first or last
% grid search for parameters: support up to 2 varing parameters
params_A = 10.^[-5:4];
epsilon = [0.1 1]; max_iter = 300; bias = 0;
if ~allframes
    for n = 1:numel(data)
        labels{n}(1,:) = src.intensity{n}(1,:);
        labels{n}(2,2) = max(src.intensity{n}(:,2));  labels{n}(2,1) = find(src.intensity{n}(:,2)==labels{n}(2,2),1,'first'); % 'last's
        labels{n}(3,:) = src.intensity{n}(end,:);
    end
else
    labels = src.intensity;
end
%%
for oter = 1:numel(params_A)
for iter = 1:length(src.idx_cv)
    inst_train = src.idx_cv(iter).train;
    inst_test = src.idx_cv(iter).validation;
    % SVR formulation: regression loss only with regularization
    N = length(inst_train);
    train_data = [];
    train_label = [];
    for n = 1:N
        train_data = [train_data; data{inst_train(n)}(:,labels{inst_train(n)}(:,1))'];
        train_label = [train_label; labels{inst_train(n)}(:,2)];
    end
    % optional scaling may be performed
    if scaled
        scale_max = max(train_data);
        scale_min = min(train_data);
        temp = bsxfun(@minus, train_data, scale_min); %data_together % ATTENTION: NOT data_together
        train_data_scaled = bsxfun(@rdivide, temp, scale_max-scale_min); %train_data % try a different way of scaling, scale train data first then use the same number to scale test data
    else
        train_data_scaled = train_data;
    end
    
    % define initial parameter of regression model
    if solver == 1        % solver: libsvm
        svm_param = [3 0 1 0.1]; % L2-regularized hinge-loss, linear kernel, gamma coefficient for kernel, tolerence 0.1
        configuration = sprintf('-s %d -t %d -g %f -c %f',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
        model = svmtrain(train_label, sparse(train_data_scaled),configuration);
        w = model.SVs' * model.sv_coef; b = -model.rho;    theta = [w(:); b];
    elseif solver == 2       % solver: liblinear
        svm_param = [11 params_A(oter) epsilon(1) bias]; % L2-regularized L2-loss(11,12) or L1-loss(13), cost coefficient 1, tolerance 0.1, bias coefficient 1
        configuration = sprintf('-s %d -c %f -p %f -B %d -q',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
        model = train(train_label, sparse(train_data_scaled), configuration);
        theta = model.w(:);
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
        test_data = bsxfun(@rdivide, test_data, scale_max'-scale_min');
    end
    dec_values = theta'*[test_data; ones(1,size(test_data,2))]; %
    RR = corrcoef(dec_values,test_label);  ry = RR(1,2);
    e = dec_values - test_label;
    mse = e(:)'*e(:)/length(e);
    abs_fold(iter,oter) = sum(abs(e))/length(e);
    ry_fold(iter,oter) = ry;
    mse_fold(iter,oter) = mse;
%     display(sprintf('validation iteration %d completed',iter));

end
display(sprintf('--grid %d completed',oter))
end  % cross-validation
time_validation = toc(tt);

%% re-train model and test on testing data 
% identify the best model parameter
tt = tic;
if iter == 1
    [~,opt] = max(ry_fold); % min(abs_fold); % 
else
    [~,opt] = max(mean(ry_fold)); %min(abs_fold); %  or mse_fold
end
params_A(opt)
% retrain model using training + validation data
for iter = 1:length(src.idx_test)
    inst_train = src.idx_test(iter).train;
    inst_test = src.idx_test(iter).validation;    %
    N = length(inst_train);
    train_data = [];
    train_label = [];
    for n = 1:N
        train_data = [train_data; data{inst_train(n)}(:,labels{inst_train(n)}(:,1))'];
        train_label = [train_label; labels{inst_train(n)}(:,2)];
    end
    if scaled
        scale_max = max(train_data);
        scale_min = min(train_data);
        temp = bsxfun(@minus, train_data, scale_min); %data_together % ATTENTION: NOT data_together
        train_data_scaled = bsxfun(@rdivide, temp, scale_max-scale_min); %train_data % try a different way of scaling, scale train data first then use the same number to scale test data
    else
        train_data_scaled = train_data;
    end
    if solver == 1        % solver: libsvm
        svm_param = [3 0 1 0.1]; % L2-regularized hinge-loss, linear kernel, gamma coefficient for kernel, tolerence 0.1
        configuration = sprintf('-s %d -t %d -g %f -c %f',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
        model = svmtrain(train_label, sparse(train_data_scaled),configuration);
        w = model.SVs' * model.sv_coef; b = -model.rho;    theta = [w(:); b];
    elseif solver == 2       % solver: liblinear
        svm_param = [11 params_A(opt) epsilon(1) bias]; % L2-regularized L2-loss(11,12) or L1-loss(13), cost coefficient 1, tolerance 0.1, bias coefficient 1
        configuration = sprintf('-s %d -c %f -p %f -B %d -q',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
        model = train(train_label, sparse(train_data_scaled), configuration);
        theta = model.w(:);
    end
    
    % perform testing
    test_data = [];
    test_label = [];
    for n = 1:length(inst_test)
        test_data = [test_data data{inst_test(n)}];
        test_label = [test_label src.intensity{inst_test(n)}(:,2)']; % intensity
    end
    if scaled
        test_data = bsxfun(@rdivide, test_data, scale_max'-scale_min');
    end
    dec_values =theta'*[test_data; ones(1,size(test_data,2))];
    RR = corrcoef(dec_values,test_label);  ry_test(iter) = RR(1,2);
    e = dec_values - test_label;
    abs_test(iter) = sum(abs(e))/length(e);
    mse_test(iter) = e(:)'*e(:)/length(e);
    time = toc(tt);
    display(sprintf('testing iteration %d completed',iter));
    
    subplot(ceil(numel(src.idx_test)/5),5,iter)
    plot(test_label); hold on; 
    plot(dec_values,'r');
    axis([0 length(dec_values) -5 9])
end
display('testing completed');

%% plot concatenate seq
% if solver == 3
%     subplot(2,1,1)
%     loglog(1:history.iter,history.s_norm,1:history.iter,history.eps_dual,'r'); title('dual feasibility')
%     subplot(2,1,2)
%     loglog(1:history.iter,history.r_norm,1:history.iter,history.eps_pri,'r'); title('primal feasibility')
% end

%% save results
mean(ry_test)
mean(mse_test)
mean(abs_test)
save(sprintf('McMaster/results/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d.mat',method,solver,scaled,allframes,option,bias), ...
    'theta','ry_test','mse_test','abs_test','ry_fold','mse_fold','abs_fold','time','time_validation','solver','scaled','allframes','params_A','inst_train','inst_test','bias','svm_param');
