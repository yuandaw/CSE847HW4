%% load data;
spam_data = load('data.txt');
labels = load('labels.txt');
%% training data setup;
train_data = spam_data(1:2000,:);
train_data = [train_data, ones(size(train_data,1),1)];
train_labels = labels(1:2000,:);
[N,D] = size(train_data);
%% test data setup;
test_data = spam_data(2001:4601,:);
test_data = [test_data,ones(size(test_data,1),1)];
test_labels = labels(2001:4601,:);
%% parameters
epsilon = 1e-5;
maxiter = 1000;
w_0 = zeros(D,1); %Initialize w
%% different training sizes
train_size = [200, 500, 800, 1000, 1500, 2000];
train_accuracy = [];
test_accuracy = [];
%% Experiment
for i = 1:length(train_size)
    %Train
    weight = logistic_train(train_data(1:train_size(i),:), train_labels(1:train_size(i),:), epsilon, maxiter,w_0);
    score_train=train_data(1:train_size(i),:)*weight;
    prediction_train = (sign(score_train)+1)/2;
    train_ac = length(find(prediction_train==train_labels(1:train_size(i),:)))/train_size(i);
    train_accuracy(:,i) = train_ac;
    %Test
    score_test=test_data*weight;
    prediction_test = (sign(score_test)+1)/2;
    test_ac = length(find(prediction_test==test_labels))/size(test_data,1);
    test_accuracy(:,i)=test_ac;
end
%% Figure
figure(1);
plot(train_size, test_accuracy, '-o', 'LineWidth', 3,'Color','red');
grid on;
set(gca,'FontSize',20);
xlabel('Training Samples','fontsize',20);
ylabel('Test Accuracy','fontsize',20);
%% sigmoid function
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
%% Training function
function weight = logistic_train(data, labels, epsilon, steps, w_0)
    [N,d] = size(data);
    weight = w_0;
    lr = 0.8;
    loss_0 = 0; 
    for j = 1:steps
        gradient = -transpose(data)*(labels - sigmoid(data * weight))/N;
        weight = weight - lr * gradient; %gradient descent
        loss = -(labels.*log(sigmoid(data * weight)) + (1- labels).*(1 - log(sigmoid(data * weight))))/N; %loss function
        if j < steps & abs(loss)<epsilon
            break;
        end
    end
end
