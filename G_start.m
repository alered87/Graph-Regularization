function G = G_start(varargin)
% G_start : generate and save options and variables of the model in G
%     
%      G = G_start()
%      G = G_start('Param1','StringValue1','Param2',NumValue2)
%
%      G: a structure with a wide set of fields, representing options to
%         compute the model and variables to save data
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it

% options default values
G.options = struct('parameters',[10;4;5],...% model parameter, see help of BuildingMatrix
                   'tau',2,... % updatng time sampling step
                   'lambda',.01,... % regularization parameter
                   'eta',.5,...% balancing between temporal and spatial contribution
                   'rho',5,... % number of neighbors to consider to compute spatial contribution
                   'epsilon',3,... % radius of the spheres in the graph
                   'classes',2,... % number of classes to predict
                   'NN',50,... Number of nearest neighbor to compute the adjacency distance matrix
                   'epsilon_distance','euclidean',... % function evaluating the distance to add a new node
                   'graph_weights','gaussian',...% weights of the spatial adjacency matrix
                   'graph_weights_parameter',5,...% parameter of the spatial weights
                   'f_bound',10); % upper bound to f
               
numberargs = nargin;

if rem(numberargs,2) ~= 0
    error('Arguments must occur in name-value pairs.');
end
if numberargs > 0
    for i = 1:2:numberargs
        if ~ischar(varargin{i})
            error('Arguments name must be strings.');
        end
        [valid, errmsg] = checkfield(varargin{i},varargin{i+1});
        if valid
            G.options.(varargin{i}) = varargin{i+1};
        else
            error(errmsg);
        end
    end
end
 

%     Variables initialization

% Linear System Varibles
[G.A,G.B,G.options.theta,G.options.order,...
               G.options.solutions] = BuildingMatrix(G.options.parameters);
G.M = expm(G.A*G.options.tau); % linear system matrix updating
G.M2 = expm(G.A*G.options.tau/2); %linear system matrix updating(half step)
G.M2Bl = G.M2*G.B/G.options.lambda; % coefficients matrix

% Graph Variables
G.f = zeros(G.options.order,G.options.classes); % Initial Cauchy Conditions
G.x = []; % last point seen
G.E = zeros(1,G.options.classes); % global error
G.f_G = zeros(1,G.options.classes+1);%number_of_nodes-by-(classes+1) matrix 
%                                    containing the last prediction on each 
%                                    node, the first element of each row 
%                                    represent the supervisions counter

G.A_G = sparse(1,1); % sparse initialization of the distance weights 
%                      adjacency matrix
G.T_G= sparse(1,1);% sparse initialization of the temporal links adjacency
%                  matrix
G.ln = 1; % pointer to last node

G.f_flag = 0; % divergence flag for f

G.f_plot = zeros(1,G.options.classes); % function saving the evolutions of 
%                                        f in time (saved for analisys)
G.step = 1; % index for f_plot
G.SE = []; % vector containing the square error on each supervised samples 
%            arrrived to check performance of learning in time 
G.Accuracy = []; % the same as SE for classification performance


end


function [valid,errmsg] = checkfield(field,value)
% {checkfield} checks validity of structure field contents.

valid = 1;
errmsg = '';

if isempty(value)
    return
end

isString = isa(value, 'char');
range = [];
requireInt = 0;
requireScalar = 0;
requireString = 0;

switch field
    case 'tau'
        requireScalar = 1;
        range = [eps Inf];
    case 'lambda'
        requireScalar = 1;
    case 'eta'
        requireScalar = 1;
        range = [0 1];         
    case 'rho'
        requireInt = 1;
        requireScalar = 1;
        range = [0 Inf];        
    case 'epsilon'
        requireScalar = 1;
        range = [eps Inf];
    case 'classes'
        requireInt = 1;
        requireScalar = 1;
        range = [1 Inf];
    case 'NN'
        requireInt = 1;
        requireScalar = 1;
        range = [0 Inf];
    case 'epsilon_distance'
        requireString = 1;
    case 'graph_weights'
        requireString = 1;
    case 'graph_weights_parameter'
        requireScalar = 1;
    case 'f_bound'
        requireScalar = 1;
    otherwise
        valid = 0;
        errmsg = ['Unknown field ' field ' for options structure.'];
end

if valid==1 && requireString && ~isString
    valid = 0;
    errmsg = (['Invalid value for' field ...
               ' parameter: Must be a string.']);
end

if valid==1 && requireScalar && isString
    valid = 0;
    errmsg = (['Invalid value for' field ...
               ' parameter: Must be a scalar.']);    
end

if valid==1 && requireInt && ((value-round(value))~=0)
    valid = 0;
    errmsg = (['Invalid value for' field ...
               ' parameter: Must be integer.']);
end

if valid==1 && ~isempty(range),
    if (value<range(1)) || (value>range(2))
        valid = 0;
        errmsg = sprintf('Invalid value for %s parameter: ', field);
        errmsg = strcat(errmsg, ...
                        sprintf('Must be in the range [%g..%g]', ...
                        range(1), range(2)));
    end
end


end

