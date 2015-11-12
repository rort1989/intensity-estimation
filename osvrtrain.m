% svr with pairwise ordinal constraintss
% formulation is described in 'formulation.pdf: SVM based formulation'
function [w, b, alpha] = osvrtrain(labelset, dataset, epsilon, gamma, option) % , kernel
if nargin < 5
    option = 1; % decide loss function type: 1. Hinge loss on both regression and ordinal; 2. Square loss on regression and hinge loss on ordinal
end
% data is stored as D*T matrix where T is length and D is feature dimension
if ~iscell(dataset) % only one sequence
    datacells{1} = dataset; 
    labelcells{1} = labelset;
else
    datacells = dataset;
    labelcells = labelset;
end
N = numel(datacells); T = zeros(N,1);
% pre-compute the needed storage
num_pairs = 0;
num_intensity = 0;
for n = 1:N
    [M,T(n)] = size(datacells{n});    
    num_pairs = num_pairs + T(n)*(T(n)+1)/2;%nchoosek(T(n),2)
    num_intensity = num_intensity + 2*size(labelcells{n},1);
end
% pre-allocate storage
A = zeros(M,num_intensity+num_pairs);
f = -epsilon(2)*ones(num_intensity+num_pairs,1);

% Form A matrix and H = A'*A and f vector
% collect data from the labeled frames
idx_cols = 0;
for n = 1:N
    labels = labelcells{n};
    data = datacells{n};    
    for i = 1:size(labels,1)
        idx_cols = idx_cols + 1;
        A(:,idx_cols) = -data(:,labels(i,1));
        f(idx_cols) = epsilon(1) + labels(i,2);
        A(:,idx_cols+num_intensity/2) = data(:,labels(i,1));
        f(idx_cols+num_intensity/2) = epsilon(1) - labels(i,2);
    end
end
%~ debug
if idx_cols ~= num_intensity/2
    error('number of labeled frames mismatch');
end
% collect data from all pairs
idx_cols = num_intensity;
for n = 1:N
    labels = labelcells{n};
    data = datacells{n};    
    [~,idx] = max(labels(:,2)); % index of apex frame
    apx = labels(idx,1);
    if apx > 1
        pairs_1 = nchoosek(apx:-1:1,2);
        for p = 1:size(pairs_1)
            idx_cols = idx_cols + 1;
            %~ debug
            if sum(A(:,idx_cols)) ~= 0
                error('idx wrong');
            end
            A(:,idx_cols) = data(:,pairs_1(p,1))-data(:,pairs_1(p,2));
        end
    end
    if apx < T(n)
        pairs_2 = nchoosek(apx:T(n),2);  % allocate space for pairs first when scale up
        for p = 1:size(pairs_2,1)
            idx_cols = idx_cols + 1;
            %~ debug
            if sum(A(:,idx_cols)) ~= 0
                error('idx wrong');
            end
            A(:,idx_cols) = data(:,pairs_2(p,1))-data(:,pairs_2(p,2));
        end
    end
end
%~ debug
if idx_cols  >= num_intensity+num_pairs
    error('number of cols overflow');
end
if sum(A(:,idx_cols+1)) ~= 0
    error('number of cols underflow');
end
A = A(:,1:idx_cols);
f = f(1:idx_cols);
if option == 1;
    H = A'*A;
elseif option == 2
    D = [0.25/gamma(1)*ones(1,num_intensity) zeros(1,idx_cols-num_intensity)];
    H = A'*A + diag(D);
end
Aeq = zeros(1,idx_cols); 
Aeq(1:num_intensity/2) = 1;
Aeq(1+num_intensity/2:num_intensity) = -1;
beq = 0;
LB = zeros(idx_cols,1); 
UB = ones(idx_cols,1);
if option == 1
    UB(1:num_intensity) = gamma(1)*ones(num_intensity,1); 
elseif option == 2
    UB(1:num_intensity) = Inf*ones(num_intensity,1); 
end
UB(num_intensity+1:end) = gamma(2)*UB(num_intensity+1:end);

% define quadratic programming problem
alpha = quadprog(H,f,[],[],Aeq,beq,LB,UB,zeros(idx_cols,1),'Algorithm','interior-point-convex'); 

% recover parameters
w = A*alpha;
ep = 1e-5;
if option == 1
    idx = find(alpha(1:num_intensity) > ep & alpha(1:num_intensity) < gamma(1)-ep); % replace 1 by a variable to adjust
elseif option == 2
    idx = find(alpha(1:num_intensity) > ep);
end    
idx_pos = idx(idx<=num_intensity/2);
idx_neg = idx(idx>num_intensity/2);
%~ debug
if max(idx_pos) > num_intensity/2 || min(idx_neg) < num_intensity/2
    error('idx of set wrong');
end
if option == 1
    sum_pos = sum(f(idx_pos)' + w'*A(:,idx_pos));
    sum_neg = sum(-f(idx_neg)' - w'*A(:,idx_neg));
    b = (sum_pos + sum_neg)/(length(idx_pos)+length(idx_neg));
elseif option == 2
    eta_pos = alpha(idx_pos)/2/gamma(1);
    eta_neg = alpha(idx_neg)/2/gamma(1);
    b_pos = f(idx_pos)' + w'*A(:,idx_pos) + eta_pos';
    b_neg = -f(idx_neg)' - w'*A(:,idx_neg) - eta_neg';
    b = (sum(b_pos)+sum(b_neg))/(length(b_pos)+length(b_neg));
end  

end