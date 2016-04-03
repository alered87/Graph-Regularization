function [TS, VS, TeS] = data_split(data,varargin)
% data_split : divide a dataset in Training, Validation and Test balancing
%              the number of samples per class
%
%      [TS, VS, TeS] = data_split(data)
%      [TS, VS, TeS] = data_split(data,options)
%
%     data: N-by-(m+c) N samples, m dimension, c classes, target NaN/Inf
%           means unlabeled
%
%     TS: Training Set
%     VS: Validation Set
%     TeS: Test Set
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it

options = struct('n_classes',10,...% target dimension
                 'TS_min',1,...%minimum number of elements in TS
                 'VS_min',0,...%minimum number of elements in VS
                 'TeS_min',0,...%minimum number of elements in TeS
                 'TS_max',Inf,...%max number of elements in TS
                 'VS_max',Inf,...%max number of elements in VS
                 'TeS_max',Inf,...%max number of elements in TeS
                 'TS_rate',.5,...%percentage of elements in TS
                 'VS_rate',0,...%percentage of elements in VS
                 'TeS_rate',0,...%percentage of elements in TeS
                 'class_balance',true,...%remove supervisions to have
                 ...                  the same number of samples per class
                 'remove_unlabeled',false);%remove unlabeled samples 
                

% check options
if rem(nargin,2) == 0
    error('Options arguments must occur in name-value pairs.');
end
for i = 2:2:nargin
    if ~ischar(varargin{i-1})
        error('Options name must be strings.');
    end
%     [valid, errmsg] = checkfield(varargin{i},varargin{i+1});
%     if valid
        options.(varargin{i-1}) = varargin{i};
%     else
%         error(errmsg);
%     end
end

N = size(data);

set = data;

for i=1:options.n_classes % supervision positions
    sup.class{i}=find(set(:,end-options.n_classes+i)==1); 
end

% supervision number
n_sup = sum(set(isfinite(set(:,end)),end-options.n_classes+1:end));
% % % % fprintf('\n');
% % % % fprintf('Number of label: %i over %i samples \n',sum(n_sup),N(1));
for i=1:options.n_classes
% % % %     fprintf('Class %i : %i \n',i,n_sup(i));
end
% % % % fprintf('\n');
% % % % fprintf('...splitting...\n');
% % % % fprintf('\n');

n_min = min(n_sup); % size of smallest class

if n_min == 0
    pos = find(n_sup == min(n_sup));
% % % %     warning('No label for class %i',pos(1));
end

min_sup_req = options.TS_min + options.VS_min + options.TeS_min ;
if min_sup_req > n_min
% % % %     warning('Not enough label for splitting');
end

if options.class_balance % balancing the samples per class
    for i=1:options.n_classes
        surplus = length(sup.class{i})-n_min;
        if surplus > 0
            p = randperm(length(sup.class{i}),n_min);
            sup.class{i} = sup.class{i}(p);
        end
        
    end
    n_sup(1:options.n_classes)=n_min;
end

% number of samples in each set
n_TS = fix(n_sup*options.TS_rate);
if any(n_TS < options.TS_min)
% % % %     warning(['Not enough elements in Training Set,' ... 
% % % %                                      ' check TS_min and TS_rate options']);
    if options.TS_min<(min(n_sup))    
        n_TS(1:options.n_classes) = options.TS_min;
    end
end
n_VS = fix(n_sup*options.VS_rate);
if any(n_VS < options.VS_min)
% % % %     warning(['Not enough elements in Validation Set,'...
% % % %                                      ' check VS_min and VS_rate options']);
    if options.VS_min<(min(n_sup))
        n_VS(1:options.n_classes) = options.VS_min;
    end
end
n_TeS = fix(n_sup*options.TeS_rate);
if any(n_TeS < options.TeS_min)
% % % %     warning(['Not enough elements in Test Set,'...
% % % %                                     'check TeS_min and TeS_rate options']);
    if options.TeS_min<(min(n_sup))
        n_TeS(1:options.n_classes) = options.TeS_min;
    end
end

% check max
n_TS = min(n_TS,options.TS_max);
n_VS = min(n_VS,options.VS_max);
n_TeS = min(n_TeS,options.TeS_max);

% % % % fprintf('Labels after splitting:\n');
% % % % fprintf('Training Set: %i\n',sum(n_TS));
% % % % fprintf('Validation Set: %i\n',sum(n_VS));
% % % % fprintf('Test Set:  %i\n',sum(n_TeS));
% % % % fprintf('\n');
    
% random splitting
TS_pos = [];
VS_pos = [];
TeS_pos = [];

for i=1:options.n_classes
    p = randperm(n_sup(i));
    tot_class = n_TS(i) + n_VS(i) + n_TeS(i) ;
% % % %     fprintf('Elements of class %i : %i \n',i,tot_class);
% % % %     fprintf(' - Training %i (%d%%)\n',n_TS(i),options.TS_rate*100);
% % % %     fprintf(' - Validation %i (%d%%)\n',n_VS(i),options.VS_rate*100);
% % % %     fprintf(' - Test %i (%d%%)\n',n_TeS(i),options.TeS_rate*100);
% % % %     fprintf('\n');
    TS_pos = [TS_pos sup.class{i}(p(1:n_TS(i)))];
    VS_pos = [VS_pos sup.class{i}(p(n_TS(i)+1:n_TS(i)+n_VS(i)))];
    TeS_pos =[TeS_pos...
              sup.class{i}(p(n_TS(i)+n_VS(i)+1:n_TS(i)+n_VS(i)+n_TeS(i)))];
end

unsup = set;
unsup(:,end-options.n_classes+1:end)=Inf;

TS = unsup;
VS = unsup;
TeS = unsup;

if isempty(TS_pos) % saving label in Training set if not empty
    TS = [];
else
    TS(TS_pos,:) = set(TS_pos,:);
end
if isempty(VS_pos) % saving label in Validation set if not empty
    VS = [];
else
    VS(VS_pos,:) = set(VS_pos,:);
end
if isempty(TeS_pos) % saving label in Test set if not empty
    TeS = [];
else
    TeS(TeS_pos,:) = set(TeS_pos,:);
end

if options.remove_unlabeled % removing unlabeled samples
    if ~isempty(TS_pos)
        TS = TS(isfinite(TS(:,end)),:);
    end
    if ~isempty(VS_pos)
        VS = VS(isfinite(VS(:,end)),:);
    end
    if ~isempty(TeS_pos)
        TeS = TeS(isfinite(TeS(:,end)),:);
    end
end


