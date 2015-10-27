%% Script for compute regression given data and intensity label for each frame
clear all;
close all;
tt = tic;
%% load data
src = load('McMaster/standard.mat','feature','intensity','idx_cv','idx_test','dfactor'); % ,'intensity'

% define constants
data = src.feature;
labels = cell(1,numel(data));
method = 1; % 1. both regression and ordinal loss  2. regression loss only 3. ordinal loss only
solver = 3; % with method 2 or 3, can choose whether using libsvm or liblinear to solve
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 1;
option = 1;
options = optimset('GradObj','on','LargeScale','off','MaxIter',1000); theta0 = zeros(size(data{1},1)+1,1);

%% parameter tuning using validation data: things to vary: params range, scaled, bias, peak position: first or last
% grid search for parameters: support up to 2 varing parameters
[params_A,params_B] = meshgrid(10.^[-4:0],10.^[0:4]); %  -3:3  
epsilon = [0.1 1]; max_iter = 300; rho = 0.1; bias = 1;
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
    % regressin and ordinal
    N = length(inst_train);
    train_data = [];
    for n = 1:N
        train_data = [train_data data{inst_train(n)}]; % should use all frames of all sequences
    end
    if scaled
        data_scaled = cell(1);
        scale_max = max(train_data,[],2);
        scale_min = min(train_data,[],2);
        for n = 1:N
            temp = bsxfun(@minus, data{inst_train(n)}, scale_min);
            data_scaled{n} = bsxfun(@rdivide, temp, scale_max-scale_min);
        end
    else
        data_scaled = data(inst_train);
    end
    gamma = [ 1 params_A(oter) ];   lambda = params_B(oter);
    if solver == 1  % note that two gammas: the second one is regularization coefficient  
        %     [f0,g0] = regressor(theta0,data,labels,gamma); numgrad = computeNumericalGradient(@(theta) regressor(theta,data,labels,gamma), theta0); err = norm(g0-numgrad);
        [theta,f,eflag,output,g] = fminunc(@(theta) regressor(theta, data_scaled, labels(inst_train), gamma), theta0, options); % _base2
    elseif solver == 2         % note that two gammas, one for each loss term        
        [w, b, alpha] = osvrtrain(labels(inst_train), data_scaled, epsilon, gamma, option);
        theta = [w(:); b];
    elseif solver == 3 % grid search on parameters: gamma(1,2) (one for each loss term) and, fix lambda,epsilon,rho        
        [w,b,history,z] = admmosvrtrain(data_scaled, labels(inst_train), gamma, 'epsilon', epsilon, 'option', option, 'max_iter', max_iter, 'rho', rho, 'lambda', lambda,'bias',bias); %
        theta = [w(:); b];
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
    abs_fold(iter,oter) = sum(abs(e))/length(e);
    iters_fold(iter,oter) = history.iter;
    ry_fold(iter,oter) = ry;
    mse_fold(iter,oter) = mse;
    %display(sprintf('validation iteration %d completed',iter));

end
display(sprintf('--grid search %d completed',oter))
end  % cross-validation
time_validation = toc(tt);

%% re-train model and test on testing data 
% identify the best model parameter
tt = tic;
if iter == 1
        [~,opt] = max(ry_fold);%min(abs_fold);%
    else
        [~,opt] = max(mean(ry_fold)); %min(abs_fold);% or mse_fold
end
gamma = [1 params_A(opt) ]
lambda = params_B(opt)
% retrain model using training + validation data
for iter = 1:length(src.idx_test)
    inst_train = src.idx_test(iter).train;
    inst_test = src.idx_test(iter).validation;    
    N = length(inst_train);
    train_data = [];
    for n = 1:N
        train_data = [train_data data{inst_train(n)}]; % should use all frames of all sequences
    end
    if scaled
        data_scaled = cell(1);
        scale_max = max(train_data,[],2);
        scale_min = min(train_data,[],2);
        for n = 1:N
            temp = bsxfun(@minus, data{inst_train(n)}, scale_min);
            data_scaled{n} = bsxfun(@rdivide, temp, scale_max-scale_min);
        end
    else
        data_scaled = data(inst_train);
    end
    if solver == 1  % note that two gammas: the second one is regularization coefficient
        [theta,f,eflag,output,g] = fminunc(@(theta) regressor(theta, data_scaled, labels(inst_train), gamma), theta0, options);
    elseif solver == 2         % note that two gammas, one for each loss term
        [w, b, alpha] = osvrtrain(labels(inst_train), data_scaled, epsilon, gamma, option);
        theta = [w(:); b];
    elseif solver == 3 % grid search on parameters: gamma(1,2) (one for each loss term) and, fix lambda,epsilon,rho
        [w,b,history,z] = admmosvrtrain(data_scaled, labels(inst_train), gamma, 'epsilon', epsilon, 'option', option, 'max_iter', max_iter, 'rho', rho, 'lambda', lambda,'bias',bias); %
        theta = [w(:); b];
    end
    % perform testing
    test_data = [];
    test_label = [];
    for n = 1:length(inst_test)
        test_data = [test_data data{inst_test(n)}];
        test_label = [test_label src.intensity{inst_test(n)}(:,2)']; % intensity
    end
    if scaled
        test_data = bsxfun(@rdivide, test_data, scale_max-scale_min);
    end
    dec_values =theta'*[test_data; ones(1,size(test_data,2))];
    RR = corrcoef(dec_values,test_label);  ry_test(iter) = RR(1,2);
    e = dec_values - test_label;
    abs_test(iter) = sum(abs(e))/length(e);
    mse_test(iter) = e(:)'*e(:)/length(e);
    time = toc(tt);
    display(sprintf('testing iteration %d completed',iter));
end
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
mean(ry_test)
mean(mse_test)
mean(abs_test)
save(sprintf('McMaster/results/exSTD_m%d_sol%d_scale%d_all%d_opt%d_bias%d.mat',method,solver,scaled,allframes,option,bias), ...
    'theta','ry_test','mse_test','abs_test','ry_fold','mse_fold','abs_fold','iters_fold','time','time_validation','solver','scaled','allframes','params_A','params_B','gamma','inst_train','inst_test','rho','lambda','bias');
