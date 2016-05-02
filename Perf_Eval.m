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

MSE = mean((0.5/N(2))*sum((predictions-targets).^2));

[~,IP]=max(predictions,[],2);
[~,IY]=max(targets,[],2);
Accuracy = mean(IY==IP);


