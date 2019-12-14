import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from mlp import MLP
from util import load_pickle_file
from util import upsample_pos


num_epochs = 80
bs = 100
learning_rate = 1e-4
net = MLP()

training_data_path = './data_processed/training_data.pkl'
label_path = './data_processed/training_lbl.pkl'
data = load_pickle_file(training_data_path)
label = load_pickle_file(label_path)
y = np.array(label)
x = np.array(data)
x, y, x_test, y_test = upsample_pos(x, y, upsample=False)
num_bs = len(x) // bs
criterion = nn.CrossEntropyLoss(weight=torch.tensor(np.array([1., 1.])).float())  
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
for epoch in range(num_epochs):
    for ii in range(num_bs - 1):  
        # Convert torch tensor to Variable
        curr_data = x[ii * bs: (ii + 1) * bs]
        curr_labels = y[ii * bs: (ii + 1) * bs]
        # print(curr_data.shape)
        # print(type(curr_data.view(-1, 651)))
        curr_data = Variable(torch.tensor(curr_data).float())
        curr_labels = Variable(torch.tensor(curr_labels).long())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(curr_data)
        loss = criterion(outputs, curr_labels)
        loss.backward()
        optimizer.step()
        
        if (ii+1) % 100 == 0:
            _, predicted = torch.max(outputs.data, 1)
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, ii+1, len(x)//bs, loss.data))
            print("train acc:" + str(1 - len(torch.nonzero(predicted - curr_labels)) * 1.0 / len(predicted)))
            test_data = Variable(torch.tensor(x_test).float())
            test_labels = Variable(torch.tensor(y_test).long())
            outputs = net(test_data)
            _, predicted = torch.max(outputs.data, 1)
            print("test acc:" + str(1 - len(torch.nonzero(predicted - test_labels)) * 1.0 / len(test_labels)))
            print("f1: " + str(f1_score(test_labels, predicted > 0.5, average=None)))
            print("precision: " + str(precision_score(test_labels, predicted > 0.5, average=None)))
            print("recall: " + str(recall_score(test_labels, predicted > 0.5, average=None)))


# Test the Model
# correct = 0
# total = 0
test_data = Variable(torch.tensor(x_test).float())
test_labels = Variable(torch.tensor(y_test).long())
outputs = net(test_data)
_, predicted = torch.max(outputs.data, 1)
# print("test acc:" + str(1 - len(torch.nonzero(predicted - test_labels)) * 1.0 / len(test_labels)))
# print("f1: " + str(f1_score(test_labels, predicted > 0.5, average="weighted")))
# print("precision: " + str(precision_score(test_labels, predicted > 0.5, average="weighted")))
# print("recall: " + str(recall_score(test_labels, predicted > 0.5, average="weighted")))
# print("f1: " + str(f1_score(test_labels, predicted < 0.5, average="weighted")))
# print("precision: " + str(precision_score(test_labels, predicted < 0.5, average="weighted")))
# print("recall: " + str(recall_score(test_labels, predicted < 0.5, average="weighted")))

# for images, labels in test_loader:
#     images = Variable(images.view(-1, 28*28)).
#     outputs = net(images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted.cpu() == labels).sum()