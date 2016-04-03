function [S,visits_counter] = random_walk(Data,total_steps,d,s,sigma)
%random_walk: generating a random wal
%
%     [S,visits_counter,f] = random_walk(Data,total_steps,d,s,sigma)
%
%     Data: number_of_samples-by-size_of_data matrix of data
%     total_steps: desired number of steps of the sequence
%     d: probability of selecting the next point by computing the  distance
%     s: number of steps avoiding to come back to a visited point
%     sigma: variance of the Gaussian distance
% 
%     S: vector containing positions in Data of the sequence
%     visits_counter: number_of_samples-by-2 matrix containing for each  
%                     sample in Data the last step of visit and the total 
%                     count of visits
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it
tic;

N = size(Data);
S(1:total_steps) = 0;
visits_counter = zeros(N(1),2);
step = s+1; % resize step to start algorithm

n = randi([1 N(1)],1,1); % starting point
S(1) = n;
visits_counter(n,:)=[step,1];

fprintf('Steps left:       ');
while step <= total_steps+s
    fprintf('\b\b\b\b\b\b\b%7i',total_steps+s-step);
    step = step+1;
    n = random_point(Data,n,visits_counter,step,d,s,sigma);
    S(step-s) = n;
    visits_counter(n,:)=[step,visits_counter(n,2)+1];
end
fprintf('\n');

% resizing steps counter
visits_counter(:,1) = max(zeros(N(1),1),visits_counter(:,1)-s); 

toc;
end


function n = random_point(Data,n,visits_counter,step,d,s,sigma)
%random_point: selecting the next point by randomly deciding between
%              invoking {follow_link} (with probability 'd') or randomly 
%              selecting a point

p = rand();

if p < d
    n = follow_link(Data,n,visits_counter,step,s,sigma,simply);
else
    range = [1 size(Data,1)];
    r = randi(range,1,1);
    n = r;
end

end


function n = follow_link(Data,n,visits_counter,step,s,sigma)
%follow_link: random selecting the following point by a probability score
%             based on Euclidean distance
%
%     n: index of selected point

avail_pos = find(step-visits_counter(:,1) > s);% avalaible nodes positions
n_choice = size(avail_pos,1);

% calculating the Gaussian distances of each point from the current point
diff=(repmat(Data(n,:),n_choice,1)-Data(avail_pos,:))';
diff=sqrt(sum(diff.^2));

Gauss_dist = exp((-diff.^2)/(2*sigma^2))/sum(exp((-diff.^2)/(2*sigma^2)));

% sorting the distances to speed-up the searching
[weights,pos] = sort(Gauss_dist);

% selection
p = rand();
prob = weights(1);
index = 1;
while p > prob
    index = index + 1;
    prob = prob + weights(index);
end

n = avail_pos(pos(index));
end


