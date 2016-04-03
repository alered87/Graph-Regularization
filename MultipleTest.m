function T = MultipleTest()
% MultipleTest: generating test for different parameters and sequences
%
%     T = MultipleTest()
%
% final order : short-medium-long training, short-long test, supervisions
%               rate, eta
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it

load('SequencesSeries01.mat');
load('Data01.mat');

% parameters
lambda = .004;
eta = [0 .5];
rho = 10;
epsilon = 2.5;
classes = 2;
tau = 2;

% varying the percentage of supervision (0 means 1 label per class) 
TS_rate = [0,.001,.005,.01,.05,.1,.5,1];

tot_train = 1;
tot_test = 2;
tot_seq_train = 3;
tot_seq_test = 2;
tot_sets = 3;
% saving sequences
for i = 1:tot_seq_train    
%     train_seq(1).Seq(i).S = ShTr01(i).S;
%     train_seq(1).Seq(i).vc = ShTr01(i).vc;
%     train_seq(2).Seq(i).S = MeTr01(i).S;
%     train_seq(2).Seq(i).vc = MeTr01(i).vc;
%     train_seq(3).Seq(i).S = LoTr01(i).S;
%     train_seq(3).Seq(i).vc = LoTr01(i).vc;

    train_seq(1).Seq(i).S = ShTr01(i).S;
    train_seq(1).Seq(i).vc = ShTr01(i).vc;
%     train_seq(1).Seq(i).S = MeTr01(i).S;
%     train_seq(1).Seq(i).vc = MeTr01(i).vc;
%     train_seq(1).Seq(i).S = LoTr01(i).S;
%     train_seq(1).Seq(i).vc = LoTr01(i).vc;

end
for i = 1:tot_seq_test
    test_seq(1).Seq(i).St = ShTe01(i).S;
    test_seq(1).Seq(i).vc = ShTe01(i).vc;
    test_seq(2).Seq(i).St = LoTe01(i).S;
    test_seq(2).Seq(i).vc = LoTe01(i).vc;
end 

Tab_size = tot_train*tot_test*length(TS_rate)*length(eta);
T = zeros(Tab_size,4);
tr_ind = 1;
tot_iter = Tab_size*tot_seq_train*tot_seq_test*tot_sets;

for num_train = 1:tot_train % number of different size in training sequences
    fprintf('Train length: %i/%i \n',num_train,tot_train);
    for num_seq_train = 1:tot_seq_train % number of sequences of each type
        fprintf('  Sequence of Train: %i/%i\n',num_seq_train,tot_seq_train);
        VisCount = train_seq(num_train).Seq(num_seq_train).vc; % saving visits counter
        Data = D01;
        Data(:,end-1:end) = Inf;
        Data(VisCount(:,2)>0,:) = D01(VisCount(:,2)>0,:);
        for n_sup = 1:length(TS_rate) % number of percentage of supervision
            fprintf('    Supervision: %1.3f\n',TS_rate(n_sup));
                for n_set = 1:tot_sets % number of random selection of supervised point
                    fprintf('      Set:  %i /%i\n ',n_set,tot_sets);
                    [TS,~,~] = data_split(Data,'n_classes',classes,'TS_rate',TS_rate(n_sup));
                    for num_eta = 1:length(eta)
                        fprintf('      eta: %1.1f \n',eta(num_eta));
                        fprintf('         Training...');
                        G = G_start('tau',tau,'lambda',lambda,'eta',eta(num_eta),'rho',rho,'classes',classes,'epsilon',epsilon);
                        G = G_train(G,TS(train_seq(num_train).Seq(num_seq_train).S,:),1);
                        for num_test = 1:tot_test % number of different size in test sequences
                           fprintf('         Test length: %i/%i \n',num_test,tot_test);
                           ind = (num_train-1)*length(TS_rate)*length(eta)*tot_test+...
                                  (num_test-1)*length(TS_rate)*length(eta)+...
                                  length(TS_rate)*(num_eta-1)+...
                                  +n_sup;                        
                           for num_seq_test = 1:tot_seq_test % number of sequences of each type
                                fprintf('           Test Sequences: %i/%i  :  ',num_seq_test,tot_seq_test);
                                fprintf('(Trial %i /%i)\n',tr_ind,tot_iter);
                                fprintf('                     Testing...');
                                [~,P] = G_test(G,Test01,test_seq(num_test).Seq(num_seq_test).St,1);
                                T(ind,1) = T(ind,1)+ P(2);
                                T(ind,2) = T(ind,2)+ P(4);
                                T(ind,3) = T(ind,3)+ P(6);
                                tr_ind = tr_ind + 1;
                                T(ind,4) = T(ind,4) +1;
%                                 fprintf('Saving...\n');
%                                 save('MultiTest01SML','T');
                           end
                        end
                    end
                end
        end
    end
end

T = T/(tot_seq_train*tot_seq_test*tot_sets);
fprintf('Saving...\n');
save('MultiTest01ShTr','T');
end