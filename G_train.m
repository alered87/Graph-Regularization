function G = G_train(G,Data,epochs)
% G_train : train the variables of G wrt Data for given epochs
%
%     G = G_train(G,Data,epochs)
%      
%     G : structure containing options and variables of the model 
%     Data : matrix containing row wise an element and its binary target 
%          (Nan/Inf-by-n_of_classes for target means unsupervised instance) 
%     epochs : number of times to train the model on data
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it

N = size(Data);

if isempty(G.x) % first sample for the model, variable initialization 
    if size(Data,2) <= G.options.classes
        error('Invalid Data dimension: check G.options.classes');
    else
        G.x = Data(1,1:end-G.options.classes);
        G.R_G = G.x; % initialization of the node matrix
        G.options.input_dimension = length(G.x); % saving input dimension
        G.A_G(1,1) = 1; % initialization of the spatial adjacency matrix
        G.T_G(1,1) = 1; % initialization of the temporal adjacency matrix
    end
else % checking input dimension   
    input_size = N(2)-G.options.classes;
    if G.options.input_dimension ~= input_size;
        error('Input dimension must be:%f',G.options.input_dimension);
    end
end
    
%     preallocating f_plot
totalsteps = size(Data,1)*epochs ;
G.f_plot = [G.f_plot; zeros(2*totalsteps,G.options.classes)] ;

fprintf('Steps left:       ');
for i = 1:epochs
    for j = 1:N(1)
        fprintf('\b\b\b\b\b\b\b %6i',totalsteps-((i-1)*N(1)+j));
        input = Data(j,1:end-G.options.classes);
        target = Data(j,1+end-G.options.classes:end);
        G = G_update(G,input,target);
        if G.f_flag > 0 % f divergence checking
            return
        end
    end
end
fprintf('\n');

end

function G = G_update(G,input,target)
% G_update : updates the variables of G wrt the input and its target
%
%     G = G_update(G,input,target)
%      
%     G : structure containing options and variables of the model 
%     input : vector of input for f 
%     target : vector of target for f


dx = (input-G.x)/G.options.tau; % 1-step finite differences input derivatives
G.b1 = 1/sqrt(1+dx*dx'); % calculating the reciprocal of b

f_tmp = G.M2*G.f; % middle-step prediction (used to compute error)
G.step = G.step+1;
G.f_plot(G.step,:) = f_tmp(1,:); % saving the value

G = G_graph(G,input); % graph variables updating

G = G_error(G,f_tmp(1,:),target); % error computation

G.f = G.M*G.f + G.M2Bl*G.b1*G.E; % f updating

if any(G.f(1,:)>G.options.f_bound)
    G.f_flag = 1;
    warning('f out of bound: enlarge regularization balancing');
    return
end

if isfinite(target) % adding the provided supervision in the current node by  
%                averaging with the old values
    node_sum = G.f_G(G.ln,2:end)*G.f_G(G.ln,1); % old contribution of 
    G.f_G(G.ln,1) = G.f_G(G.ln,1)+1;% increase the node supervision counter
    G.f_G(G.ln,2:end) = (target+node_sum)/G.f_G(G.ln,1); % averaging the new y
    [SE,A] = Perf_Eval(f_tmp(1,:),target);
    G.SE(end+1) = SE;
    G.Accuracy(end+1) = A;
else % no supervision
    if G.f_G(G.ln,1) == 0
        G.f_G(G.ln,2:end) = G.f(1,:);%saving the new prediction in the node
    end
end
   
G.x = input; % saving current input for the next step derivatives approximation

G.step = G.step+1;
G.f_plot(G.step,:) = G.f(1,:); % saving the new value

end


function G = G_graph(G,x)
% G_graph : updates the variables of the graph in G wrt the current input x
%
%     G = G_graph(G,x)
%
%     G : structure containing options and variables of the model
%     x : current input

d = G_eps_distance(G,x,G.R_G(G.ln,:));

if d <= G.options.epsilon % matching with last node
    G.T_G(G.ln,G.ln) = G.T_G(G.ln,G.ln)+1; % updating temporal link
else   
    D = G_eps_distance(G,x,G.R_G);% computing the distances
    [D,P] = sort(D);
    if D(1) <= G.options.epsilon % matching best node
        tmp = G.ln;
        G.ln = P(1); % updating last node index
        G.T_G(G.ln,tmp) = G.T_G(G.ln,tmp)+1; % updating temporal link
    else % adding a new node
        G.R_G = [G.R_G;x]; % adding new node to R_G
        G.T_G(end+1,G.ln) = 1; % updating temporal link
        G.T_G(end,end+1) = 0; % resizing T_G
        G.ln = size(G.R_G,1); % updating last node index
        G.f_G(G.ln,:) = [0 G.f(1,:)]; % saving the current value of f in 
                                      % the new node
        G.A_G(end+1,end+1) = 1;        
        nn = min(length(D),G.options.NN); % number of edges to calculate in 
%                                           the adjacency matrix A_G
        if nn>0
            W = G_graph_weights(G,D(1:nn)); % calculating weights of A_G
            G.A_G(end,P(1:nn)) = W; % insert the first nn distance
        end
    end
end

end


function G = G_error(G,f,y)
% G_error : calculate the sum of the external and graph contribution to
%   update f
%
%     G = G_error(G,f,y)
%
%     G : structure containing options and variables of the model
%     f : current prediction
%     y : possible external supervision (Inf/Nan means unsupervised example

%     calculating the external supervision contribution
if isfinite( y(end) )
    G.V_y = f-y;              % supervised sample
else
    G.V_y( 1:G.options.classes ) = 0; % unsupervised sample
end

G = G_spatial_error(G,f); % spatial contribution

G = G_temporal_error(G,f); % temporal contribution


G.E = G.V_y+G.V_T+G.V_S; % global error

end


function G = G_spatial_error(G,f)
% G_spatial_error : calculating the error contribution from the nearest
% 	nodes in G to the current one(up to rho),wrt the current prediction f
%
%     G = G_spatial_error(G,f)
%
%     G : structure containing the variables of the model
%     f : current prediction

ps = nnz(G.A_G(G.ln,:)); % non-zero distance in the ln row

ns = min(G.options.rho,ps); % number of neighbors to use
if ns>0
    [W,P] = sort(G.A_G(G.ln,:)); % sorting weights

    P = P(end-ns+1:end); % taking the closest ns
    W = full(W(end-ns+1:end));


    F = repmat(f,ns,1); % calculating differences
    DF = F-G.f_G(P,2:end);

    W = repmat(W',1,G.options.classes); % weights multiplication
    G.V_S = sum(W.*DF,1);


    G.V_S = ((1-G.options.eta)/ns)*G.V_S; % normalization
else
    G.V_S(1:G.options.classes) = 0;
end
end


function G = G_temporal_error(G,f)
% G_temporal_error : calculating the error contribution from the nodes 
% 	linked to the current one by a temporal edge, wrt the prediction f
%
%     G = G_temporal_error(G,f)
%
%     G : structure containing the variables of the model
%     f : current prediction

P = find(G.T_G(G.ln,:)~=0); % finding link position in T_G

F = repmat(f,length(P),1);  % calculating differences
DF = F-G.f_G(P,2:end);

W = G.T_G(G.ln,P); % weights multiplication
W = full(W);
W = repmat(W',1,G.options.classes);
G.V_T = sum(W.*DF,1)/sum(W(:,1)); % averaged combination

norm = max(W(:,1))/(max(max(G.T_G))); % global normalization coefficient

G.V_T = G.options.eta*norm*G.V_T; % normalization

end

function W = G_graph_weights(G,D)
% G_graph_weights : computes the weights of the adjacency matrix from the
%   euclidean distances in D, according to the method in G.graph_weights
%
%      W = G_graph_weights(G,D)
%      
%      G : structure containing options and variables of the model 
%      D : vector of the euclidean distances 
% 
%      W : weights vector of A_G

switch G.options.graph_weights    
    case 'gaussian' 
        t = G.options.graph_weights_parameter;
        W = exp((-D.^2)/(2*t*t)); % computing the distances
% possible implementation of other cases
    otherwise
        error('Unknown graph weights method: %s',...
                                        G.options.graph_weights_parameter);
end

end


function D = G_eps_distance(G,X,Y)
% G_eps_distance : computes the distances between the elements of X and Y,
%   using the method according to G.epsilon_distance
%
%      D = G_eps_distance(G,X,Y)
%      
%      G : structure containing options and variables of the model
%      X : M-by-P matrix of M P-dimensional vectors 
%      Y : N-by-P matrix of M P-dimensional vectors
% 
%      D : M-by-N distance matrix

switch G.options.epsilon_distance
    case 'euclidean'
        D = euclidean(X,Y);
% possible implementation of other cases
    otherwise
        error('Unknown graph nodes distance method: %s',...
                                               G.options.epsilon_distance);
end

end

function D = euclidean(A,B)
% {euclidean} computes the Euclidean distance.
%
%      D = euclidean(A,B)
%      
%      A: M-by-P matrix of M P-dimensional vectors 
%      B: N-by-P matrix of M P-dimensional vectors
% 
%      D: M-by-N distance matrix
%
% Author: Stefano Melacci (2009)
%         mela@dii.unisi.it
%         * based on the code of Vikas Sindhwani, vikas.sindhwani@gmail.com

if (size(A,2) ~= size(B,2))
    error('A and B must be of same dimensionality.');
end

if (size(A,2) == 1) % if dim = 1...
    A = [A, zeros(size(A,1),1)];
    B = [B, zeros(size(B,1),1)];
end

aa=sum(A.*A,2);
bb=sum(B.*B,2);
ab=A*B';

D = real(sqrt(repmat(aa,[1 size(bb,1)]) + repmat(bb',[size(aa,1) 1]) -2*ab));
end
