clear all;
close all;
clc;

load("mnist.mat");

isPCA = 0;

lambda = 0;
lr = 1e-2;

batch_size = 256;
batch_num  = floor(training.count / batch_size);

epoch_num = 10;

hidden_num = [784,256,128];
layer = length(hidden_num) -1;

W = cell(1,2*layer);
dW = cell(1,2*layer);
In = cell(1,2*layer+1);
Out = cell(1,2*layer+1);
for i = 1:layer
    W{i}=rand(hidden_num(i+1),hidden_num(i)+1)-0.5;
    W{2*layer-i+1}=rand(hidden_num(i),hidden_num(i+1)+1)-0.5;
end

for epoch_idx = 1:epoch_num
    for batch_idx = 1:batch_num
        X = training.images(:,:,(batch_idx-1)*batch_size+1:batch_idx*batch_size);
        X = reshape(X,[28*28,batch_size]);
        
        % feed forward
        In{1} = X;
        Out{1} = [X;ones(1,batch_size)];
        for i = 2:layer*2+1
            if isPCA == 1 && i == layer+1
                In{i} = W{i-1} * Out{i-1};
                Out{i} = [In{i};ones(1,batch_size)];
            else
                In{i} = W{i-1} * Out{i-1};
                Out{i} = [(-exp(-In{i}) + exp(In{i}))./(exp(In{i}) + exp(-In{i}));ones(1,batch_size)];
            end
        end
        loss = 0.5 * mean(sum((Out{2*layer+1}(1:hidden_num(1),:) - In{1}).^2)) ...
            + lambda * 0.5 * mean(sum(Out{layer+1}(1:hidden_num(layer+1),:).^2));
        disp (loss);
        % feed back
        delta = NaN;
        for i = 2*layer : -1: 1
            if isnan(delta)
                delta = 1/batch_size .* (Out{i+1}(1:hidden_num(1),:) - In{1}) .* (1-Out{i+1}(1:hidden_num(1),:).^2);
            elseif i == layer && isPCA == 1
                delta = W{i+1}' * delta;
                delta = delta(1:size(delta,1)-1,:);
            else
                delta = W{i+1}' * delta;
                delta = delta(1:size(delta,1)-1,:) .*  (1-Out{i+1}(1:size(delta,1)-1,:).^2);
            end
            dW{i} = delta * Out{i}';
        end
        delta = NaN;
        for i = layer:-1:1
            if isnan(delta)
                if isPCA == 0
                    delta = lambda * Out{layer+1}(1:hidden_num(layer+1),:) .* (1-Out{layer+1}(1:hidden_num(layer+1),:).^2);
                else
                    delta = lambda * Out{layer+1}(1:hidden_num(layer+1),:);
                end
                delta = delta / batch_size;
            else
                delta = W{i+1}' * delta;
                delta = delta(1:size(delta,1)-1,:) .* (1-Out{i+1}(1:size(delta,1)-1,:).^2);
            end
            dW{i} = dW{i} + delta * Out{i}';
        end
        % W = W - lr * dW
        for i = 1:2*layer
            W{i} = W{i} - lr * dW{i};
        end
    end
end

% show
num_show = 5;
idx = randperm(test.count,num_show);
Img_in = test.images(:,:,idx);
X_in = reshape(Img_in,[28*28,num_show]);
In{1} = X_in;
Out{1} = [X_in;ones(1,num_show)];
for i = 2:layer*2+1
    if isPCA == 1 && i == layer+1
        In{i} = W{i-1} * Out{i-1};
        Out{i} = [In{i};ones(1,num_show)];
    else
        In{i} = W{i-1} * Out{i-1};
        Out{i} = [(-exp(-In{i}) + exp(In{i}))./(exp(In{i}) + exp(-In{i}));ones(1,num_show)];
    end
end
X_out = Out{2*layer+1}(1:hidden_num(1),:);
Img_out = reshape(X_out,[28,28,num_show]);
figure;
for i = 1:num_show
    subplot(2,num_show,i);
    imshow(reshape(Img_in(:,:,i),[28,28]));
    subplot(2,num_show,num_show+i);
    imshow(reshape(Img_out(:,:,i),[28,28]));
end