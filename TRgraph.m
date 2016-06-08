classdef TRgraph < handle
    %TRgraph : class for on-line graph regularization from data following 
    % temporal manifold
    
    properties
        parameters = [10;4;10]; % model parameter, see help of BuildingMatrix
        tau = 12; % updatng time sampling step
        lambda = .01; % regularization parameter
        eta = .5; % balancing between temporal and spatial contribution
        rho = 5; % number of neighbors to consider to compute spatial contribution
        epsilon = 3;  % radius of the spheres in the graph
        classes = 10;  % number of classes to predict
        NN = 50; % Number of nearest neighbor to compute the adjacency distance matrix
        epsilon_distance = 'euclidean'; % function evaluating the distance to add a new node
        graph_weights = 'gaussian'; % weights of the spatial adjacency matrix
        graphWeightsParam = 3; % parameter of the spatial weights
        f_bound = 10; % upper bound for the predictions
        % fixed
        x = []; % last point seen
        b; % reciprocal of square root input derivative
        A_G = sparse(1,1); % distance weights adjacency matrix
        T_G = sparse(1,1); % temporal links adjacency matrix
        R_G = sparse(1,1); % node matrix
        ln = 1; % pointer to last node
        f_flag = 0; % divergence flag for f
        step = 1; % index for f_plot
%         SE = []; % square error in epochs of training vector 
%         Accuracy = []; % classification accuracy in epochs of training vector 
        % computed
        A;B;theta;order;solutions; % Linear System Variables
        M; % linear system matrix for updating formula
        M2; % linear system matrix for half step updating formula
        M2Bl; % coefficients matrix for updating formula
        f; % Initial Cauchy Conditions
        E;V_y;V_T;V_S; % error gradients
        f_G; % number_of_nodes-by-(classes+1) matrix containing the last 
             % prediction on each node, first element of row is supervisions counter
        f_plot; % function saving the evolutions of f in time (saved for analisys)
    end
    
    methods     
%% CONSTRUCTOR
        function G = TRgraph(varargin)
            numberargs = nargin;
            if rem(numberargs,2) ~= 0
                error('Arguments must occur in name-value pairs.');
            end
            if numberargs > 0
                for i = 1:2:numberargs
                    if ~ischar(varargin{i})
                        error('Arguments name must be strings.');
                    end
                    G.(varargin{i}) = varargin{i+1};
                end
            end
            [G.A,G.B,G.theta,G.order,...
               G.solutions] = makeSystemMatrix(G.parameters);
            G.M = expm(G.A*G.tau);
            G.M2 = expm(G.A*G.tau/2);
            G.M2Bl = G.M2*G.B/G.lambda;
            G.f = zeros(G.order,G.classes);
            G.f_G = zeros(1,G.classes+1);
            G.f_plot = zeros(1,G.classes);
        end

%% Setting Methods       
        function set.tau(G,tau)
            msg = 'tau must be a real number greater than 0';
            if isa(tau,'char') || tau<=0 || ~isfinite(tau) || isempty(tau)
                error(msg);
            else
                G.tau = tau;
            end
        end
%%
        function set.lambda(G,lambda)
            msg = 'lambda must be a real number greater than 0';
            if isa(lambda,'char') || lambda<=0 || ~isfinite(lambda) || isempty(lambda)
                error(msg);
            else
                G.lambda = lambda;
            end
        end
%% 
        function set.eta(G,eta)
            msg = 'eta must be a real number between 0 and 1';
            if isa(eta,'char') || eta<0 || eta>1 || ~isfinite(eta) || isempty(eta)
                error(msg);
            else
                G.eta = eta;
            end
        end
%%       
        function set.rho(G,rho)
            msg = 'rho must be an integer';
            if ((rho-round(rho))~=0)
                error(msg);
            else
                G.rho = rho;
            end
        end
%%        
        function set.epsilon(G,epsilon)
            msg = 'epsilon must be a positive real number';
            if isa(epsilon,'char') || epsilon<0 || ~isfinite(epsilon) || isempty(epsilon)
                error(msg);
            else
                G.epsilon = epsilon;
            end
        end
%%        
        function set.classes(G,classes)
            msg = 'classes must be an integer';
            if ((classes-round(classes))~=0)
                error(msg);
            else
                G.classes = classes;
            end
        end
%%        
        function set.NN(G,NN)
            msg = 'classes must be an integer';
            if ((NN-round(NN))~=0)
                error(msg);
            else
                G.NN = NN;
            end
        end
%%       
        function set.epsilon_distance(G,epsilon_distance)
            msg = 'epsilon_distance must be a string';
            if ~isa(epsilon_distance,'char') || isempty(epsilon_distance)
                error(msg);
            else
                G.epsilon_distance = epsilon_distance;
            end
        end        
%%       
        function set.graph_weights(G,graph_weights)
            msg = 'graph_weights must be a string';
            if ~isa(graph_weights,'char') || isempty(graph_weights)
                error(msg);
            else
                G.graph_weights = graph_weights;
            end
        end
%%       
        function set.graphWeightsParam(G,graphWeightsParam)
            msg = 'graphWeightsParam must be a real number strictly greater than 0';
            if isa(graphWeightsParam,'char') || graphWeightsParam<=0 || ~isfinite(graphWeightsParam) || isempty(graphWeightsParam)
                error(msg);
            else
                G.graphWeightsParam = graphWeightsParam;
            end
        end
%%       
        function set.f_bound(G,f_bound)
            msg = 'f_bound must be a real number greater than 0 (possibly Inf/NaN)';
            if isa(f_bound,'char') || f_bound<=0 || isempty(f_bound)
                error(msg);
            else
                G.f_bound = f_bound;
            end
        end        
%%
        function D = epsDistance(G,X,Y)
        % epsDistance : computes the distances between the elements of X and Y,
        %   using the method according to G.epsilon_distance
        %
        %      D = epsDistance(G,X,Y)
        %      
        %      X : M-by-P matrix of M P-dimensional vectors 
        %      Y : N-by-P matrix of M P-dimensional vectors
        % 
        %      D : M-by-N distance matrix

        switch G.epsilon_distance
        case 'euclidean'
            D = euclidean(X,Y);
        % possible implementation of other cases
        otherwise
            error('Unknown graph nodes distance method: %s',G.epsilon_distance);
        end

        end
        
%%
        function W = graphWeights(G,D)
        % graphWeights : computes the weights of the adjacency matrix from the
        %        distances in D, according to the method in G.graph_weights
        %
        %      W = graphWeights(G,D)
        %      
        %      D : vector of the euclidean distances 
        % 
        %      W : weights vector of A_G

        switch G.graph_weights    
            case 'gaussian' 
                t = G.graphWeightsParam;
                W = exp((-D.^2)/(2*t*t)); % computing the distances
        % possible implementation of other cases
            otherwise
                error('Unknown graph weights method: %s',G.graphWeightsParam);
        end

        end
        
%%
        function temporalWeights(G,f)
        % temporalWeights : calculating the error contribution from the nodes 
        % 	linked to the current one by a temporal edge, wrt the prediction f
        %
        %     temporalWeights(G,f)
        %
        %     f : current prediction

        P = find(G.T_G(G.ln,:)~=0); % finding link position in T_G
        F = repmat(f,length(P),1);  % calculating differences
        DF = F-G.f_G(P,2:end);
        W = G.T_G(G.ln,P); % weights multiplication
        W = full(W);
        W = repmat(W',1,G.classes);
        G.V_T = sum(W.*DF,1)/sum(W(:,1)); % averaged combination
        norm = max(W(:,1))/(max(max(G.T_G))); % global normalization coefficient
        G.V_T = G.eta*norm*G.V_T; % normalization

        end
        
%%
        function spatialWeights(G,f)
        % spatialWeights : calculating the error contribution from the nearest
        % 	nodes in G to the current one(up to rho),wrt the current prediction f
        %
        %     f : current prediction

        ps = nnz(G.A_G(G.ln,:)); % non-zero distance in the ln row
        ns = min(G.rho,ps); % number of neighbors to use
        if ns>0
            [W,P] = sort(G.A_G(G.ln,:)); % sorting weights
            P = P(end-ns+1:end); % taking the closest ns
            W = full(W(end-ns+1:end));
            F = repmat(f,ns,1); % calculating differences
            DF = F-G.f_G(P,2:end);
            W = repmat(W',1,G.classes); % weights multiplication
            G.V_S = sum(W.*DF,1);
            G.V_S = ((1-G.eta)/ns)*G.V_S; % normalization
        else
            G.V_S(1:G.classes) = 0;
        end
        end
        
%%
        function error(G,f,y)
        % error : calculate the sum of the external and graph contribution to
        %   update f
        %
        %     error(G,f,y)
        %
        %     f : current prediction
        %     y : possible external supervision (Inf/Nan means unsupervised
        %         example)

        %     calculating the external supervision contribution
        if any(~isfinite(y(end)))
            G.V_y(1:G.classes) = 0; % unsupervised sample
        else
            G.V_y = f-y;              % supervised sample
        end
        G.spatialWeights(f); % spatial contribution
        G.temporalWeights(f); % temporal contribution
        G.E = G.V_y+G.V_T+G.V_S; % global error
        end
        
%%
        function epsNetUpdate(G,input)
        % graphUpdate : updates the variables of the graph in G wrt the current input x
        %
        %     graphUpdate(G,x)
        %
        %     x : current input

        d = G.epsDistance(input,G.R_G(G.ln,:));

        if d <= G.epsilon % matching with last node
            G.T_G(G.ln,G.ln) = G.T_G(G.ln,G.ln)+1; % updating temporal link
        else   
            D = G.epsDistance(input,G.R_G);% computing the distances
            [D,P] = sort(D);
            if D(1) <= G.epsilon % matching best node
                tmp = G.ln;
                G.ln = P(1); % updating last node index
                G.T_G(G.ln,tmp) = G.T_G(G.ln,tmp)+1; % updating temporal link
            else % adding a new node
                G.R_G = [G.R_G;input]; % adding new node to R_G
                G.T_G(end+1,G.ln) = 1; % updating temporal link
                G.T_G(end,end+1) = 0; % resizing T_G
                G.ln = size(G.R_G,1); % updating last node index
                G.f_G(G.ln,:) = [0 G.f(1,:)]; % saving the current value of f in 
                                              % the new node
                G.A_G(end+1,end+1) = 1;        
                nn = min(length(D),G.NN); % number of edges to calculate in the adjacency matrix A_G
                if nn>0
                    W = G.graphWeights(D(1:nn)); % calculating weights of A_G
                    G.A_G(end,P(1:nn)) = W; % insert the first nn distance G.A_G(P(1:nn),end) = W; % symmetrizing 
                end
            end
        end

        end

%%
        function update(G,input,target)
        % update : updates the variables of G wrt the input and its target
        %
        %     update(G,input,target)
        %      
        %     input : vector of input for f 
        %     target : vector of target for f

        dx = (input-G.x)/G.tau; % 1-step finite differences input derivatives
        G.b = 1/sqrt(1+dx*dx'); % calculating the reciprocal of b

        f_tmp = G.M2*G.f; % middle-step prediction (used to compute error)
        G.step = G.step+1;
        G.f_plot(G.step,:) = f_tmp(1,:); % saving the value

        G.epsNetUpdate(input); % graph variables updating
        G.error(f_tmp(1,:),target); % error computation
        G.f = G.M*G.f + G.M2Bl*G.b*G.E; % f updating

        if any(G.f(1,:)>G.f_bound)
            G.f_flag = 1;
            warning('f out of bound: enlarge regularization balancing');
            return
        end

        if isfinite(target) % adding the provided supervision in the current node by  
                            % averaging with the old values
            node_sum = G.f_G(G.ln,2:end)*G.f_G(G.ln,1); % old contribution of f
            G.f_G(G.ln,1) = G.f_G(G.ln,1)+1;% increase the node supervision counter
            G.f_G(G.ln,2:end) = (target+node_sum)/G.f_G(G.ln,1); % averaging the new y
%             [SE,A] = Perf_Eval(f_tmp(1,:),target);
%             G.SE(end+1) = SE;
%             G.Accuracy(end+1) = A;
        else % no supervision
            if G.f_G(G.ln,1) == 0
                G.f_G(G.ln,2:end) = G.f(1,:);%saving the new prediction in the node
            end
        end
        G.x = input; % saving current input for the next step derivatives approximation
        G.step = G.step+1;
        G.f_plot(G.step,:) = G.f(1,:); % saving the new value
        end
        
%%
        function train(G,Data,epochs)
        % train : train the variables of G wrt Data for given epochs
        %
        %     train(G,Data,epochs)
        %      
        %     Data : matrix containing row wise an element and its binary target 
        %          (Nan/Inf-by-n_of_classes for target means unsupervised instance) 
        %     epochs : number of times to train the model on data
        %
        % Author: Alessandro Rossi (2016)
        %         rossi111@unisi.it

        N = size(Data);

        if isempty(G.x) % first sample for the model, variable initialization 
            if size(Data,2) <= G.classes
                error('Invalid Data dimension: check G.options.classes');
            else
                G.x = Data(1,1:end-G.classes);
                G.R_G = sparse(G.x); % initialization of the node matrix
                G.A_G(1,1) = 1; % initialization of the spatial adjacency matrix
                G.T_G(1,1) = 1; % initialization of the temporal adjacency matrix
            end
        else % checking input dimension   
            input_size = N(2)-G.classes;
            if length(G.x) ~= input_size;
                error('Input dimension must be:%f',length(G.x));
            end
        end

        %     preallocating f_plot
        totalsteps = size(Data,1)*epochs ;
        G.f_plot = [G.f_plot; zeros(2*totalsteps,G.classes)] ;

        fprintf('Steps left:       ');
        for i = 1:epochs
            for j = 1:N(1)
                fprintf('\b\b\b\b\b\b\b %6i',totalsteps-((i-1)*N(1)+j));
                input = Data(j,1:end-G.classes);
                target = Data(j,1+end-G.classes:end);
                G.update(input,target);
                if G.f_flag > 0 % f divergence checking
                    return
                end
            end
        end
        fprintf('\n');
        end

%%
        function [Accuracy,MSE] = test(G,Data)
        % test: calculate performance on data by finding the nearest neighbor among the G nodes
        %
        %    [Accuracy,MSE] = test(G,Data)
        %
        %     Data : matrix containing row wise an element and its binary target 
        %          (Nan/Inf-by-n_of_classes for target means unsupervised instance)        

        Data = Data(isfinite(sum(Data(:,end-G.classes+1:end),2)),:);
        D = G.epsDistance(Data(:,1:end-G.classes),G.R_G);
        [~,I]=sort(D,2);
        P = G.f_G(I(:,1),2:end);
        Y = Data(:,end-G.classes+1:end);
        [Accuracy,MSE] = performance(P',Y');
        end
        
    end
end


function [Accuracy,MSE] = performance(Predictions,Targets)
% performance : calculate the prediction accuracy and MSE
%
%     [Accuracy,MSE] = performance(Predictions,Targets)
%
%     Predictions: output_size-by-n_of_samples matrix of predictions
%     Targets : output_size-by-n_of_samples matrix of targets 
 

Targets = Targets(:,isfinite(sum(Targets,1)));
Predictions = Predictions(:,isfinite(sum(Targets,1)));
N = numel(Targets);

if size(Targets,1)>1
    Accuracy = mean(vec2ind(Predictions)==vec2ind(Targets));
else
    Accuracy = mean((Predictions>.5)==(Targets>.5));
end
MSE = (0.5/N)*sum(sum((Predictions - Targets).^2,1),2);
end

