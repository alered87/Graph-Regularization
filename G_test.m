function [G,P] = G_test(G,TestData,sequence,epochs)
% G_test_last : evaluate the MSE and classification accuracy of the
%   predictions saved in G wrt the last occurence of each samples of 
%   TestData in the test sequence                
%
%     [G,P] = G_test(G,TestData,sequence)
%
%     G : structure containing options and variables of the model 
%     TestData : matrix containing row wise an element and its target  
%                (Nan/Inf for target means unsupervised instance)
%     sequence: data appearance order (multiple repetions assumed)
%
%     G: updated structure 
%     P: containd MSE and Prediction Accuracy calculated respectively:
%               - on data sequence
%               - by using the prediction on each sample last apparition
%               - by averaging over the repetitions on each samples
%        class determined with the max function on a binary target
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it

P = zeros(epochs,6);

unsup = TestData(sequence,:);
unsup(:,end-G.options.classes+1:end)=Inf;

targets = TestData(sequence,end-G.options.classes+1:end);

for i = 1:epochs
    fprintf('Iteration %i/%i ; ',i,epochs);
    G = G_train(G,unsup,1);
    if G.f_flag < 1
%         evaluating predictions on the sequence
    Start = size(G.f_plot,1)-(2*length(sequence))+2;
    [P(i,1),P(i,2)] = Perf_Eval(G.f_plot(Start:2:end,:),targets);
%         evaluating predictions on the last occurences of predictions
    [P(i,3),P(i,4)] = G_test_last(G,TestData,sequence);
%         evaluating the average of predictions on samples
    [P(i,5),P(i,6)] = G_test_avg(G,TestData,sequence);
    else
        return
    end
end

end


function [MSE, Accuracy] = G_test_last(G,TestData,sequence)
% G_test_last : evaluate the MSE and classification accuracy of the
%   predictions saved in G wrt the last occurence of each samples of 
%   TestData in the test sequence                
%
%     [MSE, Accuracy] = G_test_last(G,TestData,sequence)
%
%     G : structure containing options and variables of the model 
%     TestData : matrix containing row wise an element and its target  
%                (Nan/Inf for target means unsupervised instance)
%     sequence: data appearance order (multiple repetions assumed)
%
%     MSE: mean square error 
%     Accuracy: prediction accuracy, class determined with the max function
%               on a binary target

predictions(1:size(TestData,1),1:G.options.classes) = Inf;
targets = TestData(:,end-G.options.classes+1:end);

Start = size(G.f_plot,1)-(2*length(sequence));

for i = 1:length(sequence)
    predictions(sequence(i),:) = G.f_plot(Start+(2*i),:);
end

% remove possible unlabeled samples
predictions = predictions(isfinite(targets(:,end)),:);
targets = targets(isfinite(targets(:,end)),:);

% remove possible unpredicted samples
targets = targets(isfinite(predictions(:,end)),:);
predictions = predictions(isfinite(predictions(:,end)),:);

[MSE, Accuracy] = Perf_Eval(predictions,targets);

end


function [MSE, Accuracy] = G_test_avg(G,TestData,sequence)
% G_test_avg : evaluate the MSE and classification accuracy of the
%   predictions saved in G wrt TestData, by averaging the performace on 
%   each samples wrt the number of its occurrences                
%
%     [MSE, Accuracy] = G_test_avg(G,TestData,sequence)
%
%     G : structure containing options and variables of the model 
%     TestData : matrix containing row wise an element and its target  
%                (Nan/Inf for target means unsupervised instance)
%     sequence: data appearance order (multiple repetions assumed)
%
%     MSE: mean square error 
%     Accuracy: prediction accuracy, class determined with the max function
%               on a binary target


X = zeros(size(TestData,1),3); %one row for each samples contains the total 
%                               occurrences counter, the accumulation of SE 
%                               and the good predictions

targets = TestData(:,end-G.options.classes+1:end);
Start = size(G.f_plot,1)-(2*length(sequence));

for i = 1:length(sequence)
    if isfinite(TestData(sequence(i),end))
        X(sequence(i),1) = X(sequence(i),1) + 1;
        d = targets(sequence(i),:) - G.f_plot(Start+(2*i),:);
        X(sequence(i),2) = X(sequence(i),2)+((d*d')/(2*G.options.classes));
        pred = find(G.f_plot(Start+(2*i),:)==max(G.f_plot(Start+(2*i),:)));        
        sup = find(targets(sequence(i),:)==max(targets(sequence(i),:)));
        if pred(1) == sup(1)            
            X(sequence(i),3) = X(sequence(i),3) + 1;  
        end
    end
end

SE = X(:,2)./X(:,1);
MSE = mean(SE(isfinite(SE)));

acc = X(:,3)./X(:,1);
Accuracy = sum(acc(isfinite(acc)))/length(acc(isfinite(acc)));

end




