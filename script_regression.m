%% Script for compute regression given data and intensity label for each frame
clear all;
close all;
tt = tic;
% load data
src = load('McMaster/McMaster.mat','LBP_features','PCA_LBP_features','PSPI','sequence');
% second experiment: complete dataset
subjects = [1:14 16:25]; % sub15 has all 0 PSPI
inst = cell(numel(subjects),1); NN = 0;
for i = 1:numel(inst)
    inst{i} = 1:numel(src.sequence{subjects(i)});
    NN = NN + numel(inst{i});
end

% second experiment: leave one subject out
inst_select = 1:NN;
idx_cv = lot_idx(inst);
method = 1; % 1. both regression and ordinal loss  2. regression loss only 3. ordinal loss only
solver = 3; % with method 2 or 3, can choose whether using libsvm or liblinear to solve
allframes = 0; % 0: use only apex and begin/end frames in labels; 1: use all frames
scaled = 0;
% grid search for parameters: support up to 2 varing parameters
[params_A,params_B] = meshgrid(10.^[-5:0],10.^[0:4]);
for oter = 1:numel(params_A)%size(params_A,2)%
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
            data{count_inst} = src.PCA_LBP_features{subjects(s)}{inst{s}(n)}'; % features
            intensity{count_inst} = src.PSPI{subjects(s)}{inst{s}(n)}'; % pain intensity: a scalar     
            id_sub(count_inst) = s;
        end
    end
end
inst_train = inst_select(idx_cv(iter).train); 
inst_test = inst_select(idx_cv(iter).validation); % inst_train; %

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
% regressin and ordinal
N = length(inst_train);
train_data = [];
for n = 1:N
    train_data = [train_data; data{inst_train(n)}']; % should use all frames of all sequences
end
if scaled
    scale_max = max(train_data);
    scale_min = min(train_data);
    for n = 1:N
        temp = bsxfun(@minus, data{inst_train(n)}, scale_min');
        data{inst_train(n)} = bsxfun(@rdivide, temp, scale_max'-scale_min');
    end
end
if solver == 1
    % define initial parameter of regression model
    rng default;
    theta0 = 0.1*randn(fdim+1,1);%
    gamma = [0.001 0];  % the second one is regularization coefficient
    % some good config: [0.001 1000] for apex&onset only; [1 100000] for all frames
    % train regression model
    % Solving minimization problem using Matlab optimization toolbox
    options = optimset('GradObj','on','LargeScale','off','MaxIter',1000);
    [f0,g0] = regressor(theta0,data,labels,gamma);
%     numgrad = computeNumericalGradient(@(theta) regressor(theta,data,labels,gamma), theta0);
%     err = norm(g0-numgrad);
    [theta,f,eflag,output,g] = fminunc(@(theta) regressor(theta,data(inst_train),labels(inst_train),gamma), theta0, options); % _base2
    if iter == length(idx_cv)
        [ff,gg,fr,fo] = regressor(theta,data(inst_train),labels(inst_train),gamma);
    end
elseif solver == 2
    gamma = [1 0.000001]; % note that two gammas, one for each loss term
    epsilon = [0.1 1];
    option = 1;
    [w, b, alpha] = osvrtrain(labels(inst_train), data(inst_train), epsilon, gamma, option);
    theta = [w(:); b];
elseif solver == 3 % grid search on parameters: gamma(2) and lambda, fix gamma(1),epsilon,rho
    gamma = [1 params_A(oter)]; % note that two gammas, one for each loss term
    epsilon = [0.1 1];
    option = 2;    max_iter = 200; rho = 0.1; lambda = params_B(oter);
    [w,b,converge,z] = admmosvrtrain(data(inst_train), labels(inst_train), gamma, 'epsilon', epsilon, 'option', option, 'max_iter', max_iter, 'rho', rho, 'lambda', lambda); % 
    theta = [w(:); b];
    if iter == 1
%         z(z<0)=0;
%         0.5*lambda*(w')*w
%         sum(z(1:72))
%         sum(z(73:end))
    end
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
% optional scaling may be performed
if scaled
    scale_max = max(train_data); 
    scale_min = min(train_data);
    temp = bsxfun(@minus, train_data, scale_min); %data_together % ATTENTION: NOT data_together     
    data_together_scaled = bsxfun(@rdivide, temp, scale_max-scale_min); %train_data % try a different way of scaling, scale train data first then use the same number to scale test data
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
    svm_param = [13 params_A(1,oter) 0.1 1]; % L2-regularized L2-loss(11,12) or L1-loss(13), cost coefficient 1, tolerance 0.1, bias coefficient 1
    configuration = sprintf('-s %d -c %f -p %f -B %d -q',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
    model = train(train_label, sparse(train_data_scaled),configuration);
    theta = model.w(:);
end

elseif method == 3 % MAY NEED FURTHER DEBUG THIS PART
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
% optional scaling may be performed
if scaled
    scale_max = max(train_data); 
    scale_min = min(train_data); 
    temp = bsxfun(@minus, train_data, scale_min); %data_together % ATTENTION: NOT data_together    
    data_together_scaled = bsxfun(@rdivide, temp, scale_max-scale_min); %train_data % try a different way of scaling, scale train data first then use the same number to scale test data
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
    svm_param = [3 1 -1]; % L2-regularized logistic regression (0,7) or square loss (2,1) hinge loss (3), cost coefficient 1, bias coefficient -1
    configuration = sprintf('-s %d -c %f -B %d',svm_param(1),svm_param(2),svm_param(3));
    model = train(train_label, sparse(train_data_scaled),configuration);
    theta = -[model.w(:); 0];    
elseif solver == 3
    % solver: rank-SVM
    
end
end

%% test: compute the prediction intensity given testing frame and learned model
% alternative: concatenate all testing frames
test_data = [];
test_label = [];
for n = 1:length(inst_test)
    test_data = [test_data data{inst_test(n)}];
    test_label = [test_label intensity{inst_test(n)}];
end
if scaled
    test_data = bsxfun(@rdivide, test_data, scale_max'-scale_min');
end
dec_values =theta'*[test_data; ones(1,size(test_data,2))];
RR = corrcoef(dec_values,test_label);  ry = RR(1,2);
e = dec_values - test_label;
mse = e(:)'*e(:)/length(e);
[value,apex] = max(test_label);
scale = dec_values(apex)/value;
ry_fold(iter,oter) = ry;
mse_fold(iter,oter) = mse;
scale_fold(iter,oter) = scale;
display(sprintf('iteration %d completed',iter));
%% plot concatenate seq
% subplot(ceil(numel(inst)/5),5,iter)
% plot(test_label); hold on; 
% plot(dec_values,'r');
% % axis([0 length(intensity{inst_test(n)}) -5 9])

end % cross-validation
%% plot intensity
% close all;
% for n = 1:length(inst_test)
%     figure;
%     plot(intensity{inst_test(n)},'r'); hold on; 
%     plot(dec_values{n});
%     %axis([0 length(idx) -1 6])
% end
display(sprintf('--grid %d completed',oter))
end
time = toc(tt);
%% save results
save(sprintf('McMaster/results/ex2_m%d_sol%d_scale%d_lot.mat',method,solver,scaled),'theta','inst_select','idx_cv','ry_fold','mse_fold','scale_fold','dfactor','time','method','solver','scaled','allframes'); %,'gamma', 'f','eflag','output','g',,'inst_train','inst_test'
mean(ry_fold)
mean(mse_fold)
