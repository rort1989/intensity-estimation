% solving the primal problem of L2 regularized hinge or squared hinge loss linear C-SVM
% i.e.    \min_{w,b}  0.5*||w||^b + \sum_{n=1}^N max(0,1-y_n(w^T x_n + b))
% this problem can be reformulated and solved by ADMM
% i.e.    \min_{X,Z}  0.5*X^T \Lambda X + \mu^T(Z)_+
%           s.t. Z = AX + e
% Refer to Darren Rhea's note chapter 18 for details.
% Another compact summary by Rui's 'wlssvm formulation/temp.pdf'
function [w,b,history,z,loss] = admmosvrtrain(dataset, labelset, gamma, varargin)
% specify default values
epsilon = [0.1 1];
rho = 1;
lambda = 1;
max_iter = 100;
tol = 1e-4;
bias = 0;
option = 1;
ABSTOL = 1e-3;
RELTOL = 1e-3;
for argidx = 1:2:nargin-3
    switch varargin{argidx}
        case 'epsilon'
            epsilon = varargin{argidx+1};
        case 'rho'
            rho = varargin{argidx+1};
        case 'lambda'
            lambda = varargin{argidx+1};
        case 'max_iter'
            max_iter = varargin{argidx+1};
        case 'tol'
            tol = varargin{argidx+1};
        case 'bias'
            bias = varargin{argidx+1};
        case 'option'
            option = varargin{argidx+1};     
    end
end

% formalize data
[A,e,M,num_intensity,num_pairs] = formdataset(dataset,labelset,epsilon,bias);

% initialization of other components
x = zeros(M+bias,1);  % first d entries are w, last entry is b
z = zeros(num_intensity+num_pairs,1);      % alternating variables
y = zeros(num_intensity+num_pairs,1);      % multipliers
% D = lambda*eye(M+bias);% D = zeros(M+1); D(1:M,1:M) = lambda*eye(M);
mu = gamma(1)*ones(num_intensity+num_pairs,1); % change the values if you want to assign different weights to different samples
mu(num_intensity+1:end) = gamma(2)/gamma(1)*mu(num_intensity+1:end);
loss = A*x+e;

% different loss functions
loss(loss<0) = 0;
if option == 1 % L2-norm hinge loss    
    f = lambda*0.5*(x')*x + mu'*loss;%f = 0.5*x'*D*x + mu'*loss;
elseif option == 2 % L2-norm squared hinge loss    
    f = lambda*0.5*(x')*x + mu'*(loss.^2);%f = 0.5*x'*D*x + mu'*(loss.^2);
elseif option == 3 % L1-norm hinge loss
    f = lambda*sum(abs(x)) + mu'*loss;% f = sum(abs(D*x)) + mu'*loss;
elseif option == 4 % L1-norm squared hinge loss
%     loss(loss<0) = 0;
%     % change the variable and parameters for subproblem
%     Lambda = zeros(2*d+2,1); Lambda(d+2:2*d+1) = 1;  % Lambda = zeros(d+1,1); Lambda(1:d) = 1; %  in this case Lambda is a vector
%     if isempty(x0), x = zeros(2*d+2,1)+0.01; else x = [x0; abs(x0(1:end-1)); x0(end)];  end 
%     A = [A zeros(N,d+1)];
%     f = Lambda'*abs(x) + mu'*(loss.^2);  
end

% main iterations
r_norm = zeros(1,max_iter);
s_norm = zeros(1,max_iter);
eps_pri = zeros(1,max_iter);
eps_dual = zeros(1,max_iter);
obj_res = zeros(1,max_iter);
obj = zeros(1,max_iter);
x_change = zeros(1,max_iter); x_flag = 0;
% y_change = zeros(1,max_iter);
% z_change = zeros(1,max_iter);
rho_sub = rho;
for it = 1:max_iter
    % update x
    if ~x_flag % in case x converged but not the objective function
        x_old = x;
        if option == 1 || option == 2 % l2 norm regularization
            if it == 1
                H = lambda/rho*eye(M+bias) + A'*A; % (M+1)*(M+1)
                [U,S,V] = svd(H,0);
                Sinv = diag(1./diag(S));
                Hinv = V*Sinv*(U')*(A');
            end
            q = (z - y/rho - e);  % (M+1)*1
            x = Hinv*q;
        elseif option == 3 || option == 4 % l1 norm regularization
            % solve a generalized lasso using ADMM
            q = (z - y/rho - e);
            %[w, FitInfo] = lasso(A(:,1:end-1),q,'Lambda',1/rho); % /(num_intensity+num_pairs)
            %x = [w; FitInfo.Intercept];
            [x, history_l1] = admmlasso_ori(A, q, lambda/rho, rho_sub, 1, 1);
            history.iter(it) = history_l1.iter;
        end
    end
    x_change(it) = norm(x-x_old);
    if x_change(it) < tol
        x_flag = 1;
    end
    
    % update z    
    z_old = z;
    Ax = A*x;
    loss = Ax + e;
    if option == 1 || option == 3  % hinge loss
        theta =  y/rho + loss - 0.5/rho*mu;  % (num_intensity+num_pairs)*1 
        a = mu/2/rho;
        z(theta>=0) = max(theta(theta>=0)-a(theta>=0),0);
        z(theta<0) = min(theta(theta<0)+a(theta<0),0);        
    elseif option == 2 || option == 4 % squared hinge loss
        theta =  y/rho + loss; % (num_intensity+num_pairs)*1        
        z(theta>=0) = rho.*theta(theta>=0)./(rho+2*mu(theta>=0));
        z(theta<0) = theta(theta<0);
    end
    %z_change(it) = norm(z-z_old);
    
    % update y
    dy = rho*(loss - z);
    y = y + dy; % (num_intensity+num_pairs)*1
    %y_change(it) = norm(dy);    
    
    % check convergence 
    % objective convergence: check the change of objective function
    loss(loss<0) = 0;
    if option == 1        
        f_new = lambda*0.5*(x')*x + mu'*loss;%f_new = 0.5*x'*D*x + mu'*loss;
    elseif option == 2
        f_new = lambda*0.5*(x')*x + mu'*(loss.^2);%f_new = 0.5*x'*D*x + mu'*(loss.^2);
    elseif option == 3
        f_new = lambda*sum(abs(x)) + mu'*loss;%f_new = sum(abs(D*x)) + mu'*(loss);
    elseif option == 4
        f_new = lambda*sum(abs(x)) + mu'*(loss.^2);%f_new = sum(abs(D*x)) + mu'*(loss.^2);
    end
    obj(it) = f_new;
    obj_res(it) = f_new - f; 
    f = f_new;
    % use residual convergence to decide if exit early
    r_norm(it) = norm(dy/rho);
    s_norm(it)  = norm(rho*(z - z_old));    % A'*
    eps_pri(it) = sqrt(num_intensity+num_pairs)*ABSTOL + RELTOL*max([norm(Ax) norm(-z) norm(e)]);
    eps_dual(it) = sqrt(M+bias)*ABSTOL + RELTOL*norm(y); %A'* 
    if (r_norm(it) < eps_pri(it) && s_norm(it) < eps_dual(it))
         break;
    end
    if abs(r_norm(it)) < tol && abs(obj_res(it)) < tol
        break;
    end
end

w = x(1:M);
b = bias*x(M+bias);
if option ~= 3
    history.iter = it;
end
history.r_norm = r_norm(1:it);
history.s_norm = s_norm(1:it);
history.eps_pri = eps_pri(1:it);
history.eps_dual = eps_dual(1:it);
history.obj_res = obj_res(1:it);
history.obj = obj(1:it);
history.rho = rho;
history.x_change = x_change(1:it);
% history.y_change = y_change(1:it);
% history.z_change = z_change(1:it);

function [L, U] = factor(A, F, rho)
    L = chol( A'*A + rho*(F')*F, 'lower' );
    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end

function [A,e,M,num_intensity,num_pairs] = formdataset(dataset,labelset,epsilon,bias)
if ~iscell(dataset) % only one sequence
    datacells{1} = dataset;
    labelcells{1} = labelset;
else
    datacells = dataset;
    labelcells = labelset;
end
N = numel(datacells); T = zeros(N,1);
num_pairs_max = 0;
num_intensity = 0;
for n = 1:N
    [M,T(n)] = size(datacells{n});    
    num_pairs_max = num_pairs_max + T(n)*(T(n)+1)/2;
    num_intensity = num_intensity + 2*size(labelcells{n},1);
end
% initialize the components for problem
% pre-allocate storage for A and e for efficiency
A = zeros(num_intensity+num_pairs_max,M+bias);
e = ones(num_intensity+num_pairs_max,1);
idx_row_I = 0;
idx_row_P = num_intensity;
num_pairs = 0;
for n = 1:length(datacells)
    data = datacells{n};
    label = labelcells{n};  
    nframe = size(label,1);
    peak = max(label(:,2)); % index of apex frame
    idx = find(label(:,2)==peak); % all the indices with peak intensity
    apx = label(idx(max(1,ceil(length(idx)/2))),1);  
    % based on apex frame, create the ordinal set
    % number of ordinal pair
    pairs = zeros(T(n)*(T(n)+1)/2,2);
    count = 0;
    for i = apx:-1:2
        pairs(count+1:count+i-1,1) = i;
        pairs(count+1:count+i-1,2) = [i-1:-1:1]';
        count = count + i-1;    
    end
    if apx < T(n)
        for i = apx:T(n)       
            pairs(count+1:count+T(n)-i,1) = i;
            pairs(count+1:count+T(n)-i,2) = [i+1:T(n)]';
            count = count + T(n)-i;
        end
    end
    pairs = pairs(1:count,:);
    num_pairs = num_pairs + count;
    % compute objective function value and gradient of objective function
    dat = data(:,label(:,1)); % M*num_labels
    tij = data(:,pairs(:,1)) - data(:,pairs(:,2)); % M*num_pairs
    % assign values
    A(idx_row_I+1:idx_row_I+nframe,1:M) = dat';
    A(idx_row_I+1+num_intensity/2:idx_row_I+nframe+num_intensity/2,1:M) = -dat';
    A(idx_row_P+1:idx_row_P+count,1:M) = -tij';
    e(idx_row_I+1:idx_row_I+nframe) = -epsilon(1)*ones(nframe,1) - label(:,2);
    e(idx_row_I+1+num_intensity/2:idx_row_I+nframe+num_intensity/2) = -epsilon(1)*ones(nframe,1) + label(:,2);
    e(idx_row_P+1:idx_row_P+count) = epsilon(2);
    idx_row_I = idx_row_I + nframe;
    idx_row_P = idx_row_P + count;
end
A = A(1:num_intensity+num_pairs,:);
if bias % augment A for including bias term
    A(1:num_intensity/2,M+1) = 1;
    A(1+num_intensity/2:num_intensity,M+1) = -1;
end
e = e(1:num_intensity+num_pairs,:);

end

end
