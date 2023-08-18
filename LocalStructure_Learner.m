% Local causal structure learning 
function [p, c, u] = LocalStructure_Learner(filename,target)
data_path = strcat('D:/Pycharm/PycharmProjects/causal-domain-adaption/dataset/Dgbr_data/bias_rate=0.25_p=10/',filename);
if exist(data_path,'file')==0
     fprintf('current data does not exist.\n');
     return;
end
data = csvread(data_path,1,0)+1;
alg_name = "PCD_by_PCD";
% 'dis' represents discrete data, 'con' denotes continues data 
data_type = "dis";
% alpha for independence test
alpha = 0.05;
% Causal_Learner
% Result1 is learned target's parents.
% Result2 is learned target's children.
% Result3 is learned target's PC, but cannot distinguish whether they are parents or children.
% Result4 is the number of conditional independence tests
% Result5 is running time
[Result1,Result2,Result3,~,~]=Causal_Learner(alg_name,data,data_type,alpha,target);
% 
p = Result1;
c = Result2;
u = Result3;
end