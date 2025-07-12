import torch
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import copy

from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_fscore_support
import argparse


class SimpleNet(nn.Module):
    def __init__(self,n_input=40,hidden_neural=80,n_hidden_layer=3,fn='sigmoid'):
        super(SimpleNet,self).__init__()
        self.output_type='logit'
        self.fn=fn
        self.layers = nn.ModuleList()
        for i in range(n_hidden_layer):
            self.layers.append(nn.Linear(n_input,hidden_neural))
            n_input=hidden_neural

        self.output=nn.Linear(n_input,4)
    
    def forward(self,x):
        for layer in self.layers[:-1]:
            if self.fn=='sigmoid':
                x=F.sigmoid(layer(x))
            elif self.fn=='tanh':
                x=F.tanh(layer(x))
            elif self.fn=='relu':
                x=F.relu(layer(x))
            elif self.fn=='leaky':
                x=F.leaky_relu(layer(x))
        x=self.output(x)
        if self.output_type=='prob':
            x=F.softmax(x,dim=1)
        # x = x.view(x.size(0), -1)
        # x=F.linear(self.output(x))
        # x=self.model(x)
        return x
    def set_prob(self):
        self.output_type='prob'
    def set_logit(self):
        self.output_type='logit'
    
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
filename = './FS-taunao-data.xlsx'
data = pd.read_excel(filename, sheet_name='data')
thres_data = pd.read_excel(filename, sheet_name='thres')
odor_class={item:idx for idx,item in enumerate(set(thres_data["class"].to_list()))}
print(odor_class)

for idx, row in thres_data.iterrows():
    # print('%s %f'%(row['RI'],row['odor threshold (ug/kg)']))
    normalized_value = row['odor threshold (ug/kg)']
    data.loc[:, row['RI']] /= normalized_value 
        
label_mapping={'Savory(High sweet)': 0, 'Savory(Low sweet)': 1, 'Traditional(High sweet)': 2,'Traditional(Low sweet)':3}

reverse_label_mapping={v:k for k,v in label_mapping.items()}
X = data.drop(['class'],axis=1).to_numpy()
Y=np.zeros(X.shape[0])

for idx,label in enumerate(data['class'].to_numpy()):
    Y[idx]=label_mapping[label]

X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0, stratify=Y
    )
# - Training: 0.75 * 80% = 60%
# - Validation: 0.25 * 80% = 20%
X_train, X_val, Y_train, Y_val = train_test_split(
X_train_val, Y_train_val, test_size=0.25, random_state=71, stratify=Y_train_val
)


train_dataset=CustomDataset(X_train,Y_train)
val_dataset=CustomDataset(X_val,Y_val)

batch_size = 20  # You can adjust this as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
torch.manual_seed(279)

model=SimpleNet(n_input=40,hidden_neural=160,n_hidden_layer=5,fn='tanh')
model.to(device)  
loss_fn = nn.CrossEntropyLoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs=1000
best_accuracy=0
best_metrics=None
best_model=None

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_features, batch_labels in train_dataloader:
        batch_features=batch_features.to(device)
        batch_labels=batch_labels.to(device)
        optimizer.zero_grad()      
        outputs=model(batch_features)
        loss=loss_fn(outputs,batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    
    epoch_val_loss = 0.0
    model.eval()

    all_true_labels = []
    all_pred_labels = []  

    with torch.no_grad():
    
        for batch_val_features, batch_val_labels in val_dataloader:
            batch_val_features=batch_val_features.to(device)
            all_true_labels.extend(batch_val_labels.numpy())
            # print('shape ',batch_test_labels.numpy().shape," ",batch_test_labels.numpy())
            batch_val_labels=batch_val_labels.to(device)
            outputs=model(batch_val_features)
            _,preds_idx = torch.max(outputs,1)
            loss=loss_fn(outputs,batch_val_labels)
            epoch_val_loss+=loss.item()
            preds_idx=preds_idx.cpu().numpy()
            all_pred_labels.extend(preds_idx)
    
        accuracy = accuracy_score(all_true_labels, all_pred_labels)

        if accuracy>best_accuracy:
            best_accuracy=accuracy
            overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_true_labels, all_pred_labels, average='macro')
            best_metrics=pd.DataFrame({'accuracy':[accuracy],'overall_precision':[overall_precision],'overall_recall':[overall_recall],'overall_f1':[overall_f1]})
            best_model = copy.deepcopy(model)   
if best_metrics is not None:
    torch.save(best_model.state_dict(),'best_ann_model.pt')