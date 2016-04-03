function [MSE, Accuracy] = Perf_Eval(predictions,targets)
% Perf_Eval : evaluate the MSE and classification accuracy of a predictors
%
%     [MSE, Accuracy] = Perf_Eval(predictions,targets)
%
%     predictions: n_of_samples-by-classes matrix of predictions
%     targets: n_of_samples-by-classes matrix of supervision
%
%     MSE: mean square error 
%     Accuracy: prediction accuracy, class determined with the max function
%               on a binary target
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it


% remove possible unlabeled samples
predictions = predictions(isfinite(targets(:,end)),:);
targets = targets(isfinite(targets(:,end)),:);

N = size(predictions);

correct = 0;
error = 0;

for i=1:N(1)
    diff = predictions(i,:) - targets(i,:);
    error = error + (diff*diff')/(2*N(2));
    pred = find(predictions(i,:)==max(predictions(i,:)));        
    sup = find(targets(i,:)==max(targets(i,:)));
    if pred(1) == sup(1)            
        correct = correct+1;  
    end
end

MSE = error/N(1) ;
Accuracy = correct/N(1) ;


