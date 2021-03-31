%% load data
data = load('ad_data.mat');
%% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations
par = [1e-8, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
%% Experiments
number_of_features = [];
AUC = [];
for i = 1:length(par)
    [w,c] = LogisticR(X_train, y_train, par(i), opts); %weight
    number_of_features(i) = nnz(w);
    score = X_test * w + c;
    [X,Y,T,AUC(i)] = perfcurve(y_test, score, 1);
end
%% Figures
figure(1);
plot(par, number_of_features,'-o', 'LineWidth', 3,'Color','red');
grid on;
set(gca,'FontSize',20);
xlabel('l_1 Regularization Parameter','fontsize',20);
ylabel('Number of features','fontsize',20);
%%%%%%%
figure(2);
plot(par, AUC,'-s', 'LineWidth', 3,'Color','blue');
grid on;
set(gca,'FontSize',20);
xlabel('l_1 Regularization Parameter','fontsize',20);
ylabel('AUC','fontsize',20);
