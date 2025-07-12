import pandas as pd
import numpy as np



from sklearn.model_selection import train_test_split
import operator
import random
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from scipy.spatial import distance
import xgboost as xgb 

from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_fscore_support, classification_report

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import copy
import matplotlib.pyplot as plt
from itertools import combinations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
import matplotlib.lines as mlines

from datetime import datetime
import sys
from scipy.stats import mode
import glob
import os

def contrastive_loss(z, labels, margin=1.0):
    
    loss = 0.0
    count = 0
    batch_size = z.size(0)
    
    # Iterate over all pairs in the batch
    for i, j in combinations(range(batch_size), 2):
        zi, zj = z[i], z[j]
        # Determine if pair is positive (1) or negative (0)
        l_ij = 1 if labels[i] == labels[j] else 0
        
        distance = torch.norm(zi - zj)
        if l_ij:  # Positive pair
            loss += distance ** 2
        else:     # Negative pair
            loss += torch.clamp(margin - distance, min=0) ** 2
        count += 1

    return loss / count if count > 0 else loss

class EnsembleNetContrastive(nn.Module):
    def __init__(self,logit_model,svm_model,knn_model,dct_model,dnn_model,features=['mean','std','max','min','entropy'],latent_dim=2):
        super(EnsembleNetContrastive,self).__init__()
        self.logit_model=logit_model
        self.svm_model=svm_model
        self.knn_model=knn_model
        self.dct_model=dct_model
        self.dnn_model=dnn_model
        self.features=features
        n_input=5*4
        if 'mean' in features:
            n_input+=4
        if 'std' in features:
            n_input+=4
        if 'max' in features:
            n_input+=4
        if 'min' in features:
            n_input+=4
        if 'entropy' in features:
            n_input+=1

        
        self.encoder = nn.Sequential(
          nn.Linear(n_input,64),
          nn.ReLU(),
          nn.Linear(64,32),
          nn.ReLU(),
          nn.Linear(32,latent_dim),
          nn.Sigmoid()
        )

        

        if 'entropy' in features:

            self.decoder = nn.Sequential(
            nn.Linear(latent_dim,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,n_input-1),
            nn.Sigmoid()
            )
            self.entropy_out = nn.Sequential(
            nn.Linear(latent_dim,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        else: 
            self.decoder = nn.Sequential(
            nn.Linear(latent_dim,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,n_input),
            nn.Sigmoid()
            )

        self.classifier=nn.Sequential(
            nn.Linear(latent_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,4),
        )
        # number of classes
        # self.centroids=nn.Parameter(torch.nrand(4, latent_dim))       

    def forward(self,x):

        pred_classes=None
        final_features=None
        if isinstance(x,np.ndarray):
            prob_logit=self.logit_model.predict_proba(x)
            prob_svm=self.svm_model.predict_proba(x)
            prob_knn=self.knn_model.predict_proba(x)
            prob_dct=self.dct_model.predict_proba(x)

        elif isinstance(x,torch.Tensor):
            prob_logit=self.logit_model.predict_proba(x.numpy())
            prob_svm=self.svm_model.predict_proba(x.numpy())
            prob_knn=self.knn_model.predict_proba(x.numpy())
            prob_dct=self.dct_model.predict_proba(x.numpy())


        # x=torch.tensor(x, dtype=torch.float32)
        prob_dnn=self.dnn_model.predict_proba(x)

        prob_logit=torch.tensor(prob_logit,dtype=torch.float32)
        prob_svm=torch.tensor(prob_svm,dtype=torch.float32)
        prob_knn=torch.tensor(prob_knn,dtype=torch.float32)
        prob_dct=torch.tensor(prob_dct,dtype=torch.float32)

        # 1. Stack raw probabilities from each model by concatenating along the feature dimension.
        #    This results in a tensor of shape [batch_size, 5*num_classes]
        raw_probs = torch.cat([prob_logit, prob_svm, prob_knn, prob_dct, prob_dnn], dim=1)

        stacked_probs = torch.stack([prob_logit, prob_svm, prob_knn, prob_dct, prob_dnn], dim=1)
        mean_probs=std_probs=max_probs=min_probs=entropy=None
        if 'mean' in self.features:
            mean_probs=torch.mean(stacked_probs, dim=1)
        if 'std' in self.features:
            std_probs=torch.std(stacked_probs, dim=1)
        if 'max' in self.features:
            max_probs,_=torch.max(stacked_probs, dim=1)
        if 'min' in self.features:
            min_probs,_=torch.min(stacked_probs, dim=1)
        
        # To avoid log(0), we add a small epsilon value.
        # entropy shape [batch_size, 1]
        if 'entropy' in self.features:
            mean_probs=torch.mean(stacked_probs, dim=1)
            entropy = -torch.sum(mean_probs * torch.log(torch.clamp(mean_probs, min=1e-10)), dim=1,keepdim=True)
        # Reshape entropy to [batch_size, 1]
        # stacked_features = torch.cat([mean_probs, entropy], dim=1)
        
        tensor_list = [t for t in [raw_probs,mean_probs,std_probs,max_probs,min_probs,entropy] if t is not None]
        final_features = torch.cat(tensor_list, dim=1)
        # print(final_features.shape)
        latent_x=self.encoder(final_features)
        
        if 'entropy' in self.features:
            x=self.decoder(latent_x)
            entropy_recon = self.entropy_out(latent_x)
            x = torch.cat([x, entropy_recon], dim=1)
        else:
            x=self.decoder(latent_x)
        
        pred_classes=self.classifier(latent_x)
        
        return latent_x,x,pred_classes,final_features

    def encode(self,x):
        final_features=None
        if isinstance(x,np.ndarray):
            prob_logit=self.logit_model.predict_proba(x)
            prob_svm=self.svm_model.predict_proba(x)
            prob_knn=self.knn_model.predict_proba(x)
            prob_dct=self.dct_model.predict_proba(x)

        elif isinstance(x,torch.Tensor):
            prob_logit=self.logit_model.predict_proba(x.numpy())
            prob_svm=self.svm_model.predict_proba(x.numpy())
            prob_knn=self.knn_model.predict_proba(x.numpy())
            prob_dct=self.dct_model.predict_proba(x.numpy())
        
        prob_dnn=self.dnn_model.predict_proba(x)
        prob_logit=torch.tensor(prob_logit,dtype=torch.float32)
        prob_svm=torch.tensor(prob_svm,dtype=torch.float32)
        prob_knn=torch.tensor(prob_knn,dtype=torch.float32)
        prob_dct=torch.tensor(prob_dct,dtype=torch.float32)

        # 1. Stack raw probabilities from each model by concatenating along the feature dimension.
        #    This results in a tensor of shape [batch_size, 5*num_classes]
        raw_probs = torch.cat([prob_logit, prob_svm, prob_knn, prob_dct, prob_dnn], dim=1)
        stacked_probs = torch.stack([prob_logit, prob_svm, prob_knn, prob_dct, prob_dnn], dim=1)
        mean_probs=std_probs=max_probs=min_probs=entropy=None
        if 'mean' in self.features:
            mean_probs=torch.mean(stacked_probs, dim=1)
        if 'std' in self.features:
            std_probs=torch.std(stacked_probs, dim=1)
        if 'max' in self.features:
            max_probs,_=torch.max(stacked_probs, dim=1)
        if 'min' in self.features:
            min_probs,_=torch.min(stacked_probs, dim=1)
        
        # To avoid log(0), we add a small epsilon value.
        # entropy shape [batch_size, 1]
        if 'entropy' in self.features:
            mean_probs=torch.mean(stacked_probs, dim=1)
            entropy = -torch.sum(mean_probs * torch.log(torch.clamp(mean_probs, min=1e-10)), dim=1,keepdim=True)

        tensor_list = [t for t in [raw_probs,mean_probs,std_probs,max_probs,min_probs,entropy] if t is not None]
        final_features = torch.cat(tensor_list, dim=1)
        # print(final_features.shape)
        latent_x=self.encoder(final_features)
        return latent_x





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
        return x
    
    def predict_proba(self,x):
        ret_numpy_arr=False
        if isinstance(x,np.ndarray):
            x=torch.tensor(x, dtype=torch.float32)
            ret_numpy_arr=True
        for layer in self.layers[:-1]:
            if self.fn=='sigmoid':
                x=F.sigmoid(layer(x))
            elif self.fn=='tanh':
                x=F.tanh(layer(x))
            elif self.fn=='relu':
                x=F.relu(layer(x))
            elif self.fn=='leaky':
                x=F.leaky_relu(layer(x))

        x=F.softmax(self.output(x),dim=1)

        if ret_numpy_arr:
            return x.detach().numpy()
        else:
            return x
        
    def predict(self,x):
        ret_numpy_arr=False
        if isinstance(x,np.ndarray):
            x=torch.tensor(x, dtype=torch.float32)
            ret_numpy_arr=True
        for layer in self.layers[:-1]:
            if self.fn=='sigmoid':
                x=F.sigmoid(layer(x))
            elif self.fn=='tanh':
                x=F.tanh(layer(x))
            elif self.fn=='relu':
                x=F.relu(layer(x))
            elif self.fn=='leaky':
                x=F.leaky_relu(layer(x))

        x=F.softmax(self.output(x),dim=1)

        _,preds_idx = torch.max(x,1)
        
        if ret_numpy_arr:
            return preds_idx.detach().numpy()

   
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

        x=self.features[idx]
        y=self.labels[idx]
        
        return x, y

filename = './FS-taunao-data.xlsx'
data = pd.read_excel(filename, sheet_name='data')
thres_data = pd.read_excel(filename, sheet_name='thres')
odor_class={item:idx for idx,item in enumerate(set(thres_data["class"].to_list()))}
print(odor_class)

label_mapping={'Savory(High sweet)': 0, 'Savory(Low sweet)': 1, 'Traditional(High sweet)': 2,'Traditional(Low sweet)':3}

reverse_label_mapping={v:k for k,v in label_mapping.items()}
X = data.drop(['class'],axis=1).to_numpy(dtype=np.float32)
Y=np.zeros(X.shape[0],dtype=np.float32)

for idx,label in enumerate(data['class'].to_numpy()):
    Y[idx]=label_mapping[label]

np.random.seed(seed=4190)

model_logit = OneVsRestClassifier(LogisticRegression(random_state=77, solver='liblinear',max_iter=100))
model_svc = OneVsRestClassifier(SVC(kernel='linear'))
model_knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5,n_jobs=10,p=1))
model_xg = xgb.XGBClassifier()
model_dt=DecisionTreeClassifier(min_samples_leaf=15,criterion='entropy')

model_dnn=SimpleNet(n_input=40,hidden_neural=160,n_hidden_layer=5,fn='tanh')
model_dnn.eval()

run_logit=True
run_svc=True
run_knn=True
run_xg=False
run_dt=True
run_dnn=True



models={'Logistic Regression':(model_logit,run_logit,'Logistic Regression.pkl'), 
        'SVM': (model_svc,run_svc,'SVM.pkl'),
        'KNN': (model_knn,run_knn,'KNN.pkl'),
        'XGBOOST':(model_xg,run_xg,'XGBOOST.pkl'),
        "DCT":(model_dt ,run_dt,'DCT.pkl'),
        "DNN":(model_dnn,run_dnn,'best_ann_model.pt')}

# First, split the dataset into 80% train+validation and 20% test
X_train_val, X_test, Y_train_val, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0, stratify=Y
)

# - Training: 0.75 * 80% = 60%
# - Validation: 0.25 * 80% = 20%
X_train, X_val, Y_train, Y_val = train_test_split(
X_train_val, Y_train_val, test_size=0.25, random_state=71, stratify=Y_train_val
)


def train_models(args):
    new_models={}
    for name, (model,run,model_file) in models.items():
        if not run:
            continue
        if name!="DNN":
            loaded_model=joblib.load(model_file)
        elif name=="DNN":
             model.load_state_dict(torch.load(model_file))
             model.eval()
             loaded_model=model
             

        new_models[name]=loaded_model

        if name!="DNN":
            pred_val=loaded_model.predict(X_val)
            overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(Y_val, pred_val, average='macro')
            accuracy = accuracy_score(Y_val, pred_val)
            print(f'{name} acc: {accuracy} overall_precision: {overall_precision}  overall_recall: {overall_recall} overall_f1: {overall_f1}')
        
        elif name=="DNN":
            X_val_tensor=torch.tensor(X_val, dtype=torch.float32)
            # Y_val_tensor=torch.tensor(Y_val, dtype=torch.long)
            yprd_tensor=model(X_val_tensor)
            _,preds_idx = torch.max(yprd_tensor,1)
            pred_val=preds_idx.numpy()
            overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(Y_val, pred_val, average='macro')
            accuracy = accuracy_score(Y_val, pred_val)
            print(f'{name} acc: {accuracy} overall_precision: {overall_precision}  overall_recall: {overall_recall} overall_f1: {overall_f1}')
    
    ensem_net=EnsembleNetContrastive(new_models['Logistic Regression'],
                new_models['SVM'],
                new_models['KNN'],
                new_models['DCT'],
                new_models['DNN'],features=['mean','std','max','min','entropy'],latent_dim=args.latent)
    

  

    batch_size=4
    val_dataset=CustomDataset(X_val,Y_val)
    test_dataset=CustomDataset(X_test,Y_test)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
   
    num_epochs=100

    criterion_recon = nn.MSELoss()  # For reconstruction (meta features + entropy)
    criterion_class = nn.CrossEntropyLoss()  # For classification; expects raw logits & class indices
    
    optimizer = optim.Adam(ensem_net.parameters(), lr=0.001)

    lambda_recon=1
    lambda_class=1
    lambda_contrast=5

    best_accuracy=0
    best_f1=0

    for epoch in range(num_epochs):
        ensem_net.train()
        epoch_loss = 0.0
        epoch_recon_loss=0
        epoch_class_loss=0
        epoch_constrast_loss=0
        for batch_val_features, batch_val_labels in val_dataloader:
            optimizer.zero_grad() 
            latent_z,recon_features,pred_classes,final_features=ensem_net(batch_val_features)
            loss_recon=criterion_recon(final_features,recon_features)
            loss_class = criterion_class(pred_classes,batch_val_labels)
            loss_contrast = contrastive_loss(latent_z, batch_val_labels, margin=1.0)
            loss=lambda_recon*loss_recon+lambda_class*loss_class+lambda_contrast*loss_contrast
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            epoch_recon_loss+=loss_recon.item()
            epoch_class_loss+=loss_class.item()
            epoch_constrast_loss+=loss_contrast.item()

        epoch_recon_loss=lambda_recon*(epoch_recon_loss/len(val_dataloader))
        epoch_class_loss=lambda_class*(epoch_class_loss/len(val_dataloader))
        epoch_constrast_loss=lambda_contrast*(epoch_constrast_loss/len(val_dataloader))
        epoch_loss=epoch_loss/len(val_dataloader)  
        print(f'epoch {epoch} training loss: {epoch_loss:.4f} (recon: {epoch_recon_loss:.4f} class_log: {epoch_class_loss:.4f} constast_loss: {epoch_constrast_loss:.4f}) f1: {overall_f1:.4f}') 
            
    
    if args.save_last:
        print(f'Save last weigth {args.output}')
        torch.save(ensem_net.state_dict(),args.output)

def other_ensem(args):
    new_models={}
    for name, (model,run,model_file) in models.items():
        if not run:
            continue
        if name!="DNN":
            loaded_model=joblib.load(model_file)
        elif name=="DNN":
             model.load_state_dict(torch.load(model_file))
             model.eval()
             loaded_model=model
             

        new_models[name]=loaded_model

    if args.voting:

        val_scores = [accuracy_score(Y_val, m.predict(X_val)) for _,m in new_models.items()]
        #Normalize scores to sum to 1 (these will be the weights)
        weights = np.array(val_scores) / np.sum(val_scores)
        # Get predict_proba for each model on test set
        # shape: (n_models, n_samples, n_classes)
        probas = np.array([m.predict_proba(X_test) for _,m in new_models.items()]) 
        # Weighted average of probabilities
        weighted_proba = np.tensordot(weights, probas, axes=([0], [0]))  # shape: (n_samples, n_classes)
        # Final prediction = argmax over classes
        y_pred_test = np.argmax(weighted_proba, axis=1) 

        test_precision_per_class, test_recall_per_class, test_f1_per_class, _ = precision_recall_fscore_support(Y_test, y_pred_test, average=None)
        test_overall_precision, test_overall_recall, test_overall_f1, _ = precision_recall_fscore_support(Y_test, y_pred_test, average='macro')
        test_accuracy = accuracy_score(Y_test, y_pred_test)
        
        voting_result=pd.DataFrame({'name':['voting'],  
                                        'test_accuracy':[test_accuracy],'test_overall_precision':[test_overall_precision],'test_overall_recall':[test_overall_recall],'test_overall_f1':[test_overall_f1],
                                        'test_class0_precission':[test_precision_per_class[0]],'test_class0_recall':[test_recall_per_class[0]],'test_class0_f1':[test_f1_per_class[0]],
                                        'test_class1_precission':[test_precision_per_class[1]],'test_class1_recall':[test_recall_per_class[1]],'test_class1_f1':[test_f1_per_class[1]],
                                        'test_class2_precission':[test_precision_per_class[2]],'test_class2_recall':[test_recall_per_class[2]],'test_class2_f1':[test_f1_per_class[2]],
                                        'test_class3_precission':[test_precision_per_class[3]],'test_class3_recall':[test_recall_per_class[3]],'test_class3_f1':[test_f1_per_class[3]],
                            })
   
    if args.xg:
        
        model = xgb.XGBClassifier(
        num_class=4,
        objective='multi:softprob',
        n_estimators=500,          # Large number, will stop early
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='mlogloss',       # Or 'mlogloss' for multiclass
        early_stopping_rounds=10
        )

        # Train with validation
        model.fit(
        X_train, Y_train,
        eval_set=[(X_val, Y_val)],
        
        verbose=False                # Show training progress
        )

        # Evaluate on test set using best iteration
        y_pred_test = model.predict(X_test)
        test_precision_per_class, test_recall_per_class, test_f1_per_class, _ = precision_recall_fscore_support(Y_test, y_pred_test, average=None)
        test_overall_precision, test_overall_recall, test_overall_f1, _ = precision_recall_fscore_support(Y_test, y_pred_test, average='macro')
        test_accuracy = accuracy_score(Y_test, y_pred_test)
        xg_result=pd.DataFrame({'name':['xg'],  
                                            'test_accuracy':[test_accuracy],'test_overall_precision':[test_overall_precision],'test_overall_recall':[test_overall_recall],'test_overall_f1':[test_overall_f1],
                                            'test_class0_precission':[test_precision_per_class[0]],'test_class0_recall':[test_recall_per_class[0]],'test_class0_f1':[test_f1_per_class[0]],
                                            'test_class1_precission':[test_precision_per_class[1]],'test_class1_recall':[test_recall_per_class[1]],'test_class1_f1':[test_f1_per_class[1]],
                                            'test_class2_precission':[test_precision_per_class[2]],'test_class2_recall':[test_recall_per_class[2]],'test_class2_f1':[test_f1_per_class[2]],
                                            'test_class3_precission':[test_precision_per_class[3]],'test_class3_recall':[test_recall_per_class[3]],'test_class3_f1':[test_f1_per_class[3]],
                                })


        if args.voting and not args.xg:
            voting_result.to_excel('voting_result.xlsx',index=False)
            print('voting_result.xlsx written out')
        elif not args.voting and args.xg:
            xg_result.to_excel('xg_result.xlsx',index=False)
            print('xg_result.xlsx written out')
        elif args.voting and args.xg:
            df=pd.concat([voting_result,xg_result],ignore_index=False)
            df.to_excel('voting_xg_result.xlsx',index=False)
            print('voting_xg_result.xlsx written out')

if __name__=='__main__':
         
    parser=argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser_name')
    train_models_parser = subparsers.add_parser('train-models')
    train_models_parser.add_argument('--latent',type=int,default=3)
    train_models_parser.add_argument('--loss_cls',action='store_true')
    train_models_parser.add_argument('--loss_rec',action='store_true')
    train_models_parser.add_argument('--loss_contrast',action='store_true')
    train_models_parser.add_argument('--save_last',action='store_true')
    train_models_parser.add_argument('--output',type=str,default='ensem_contrasive_weight.pt')

    other_ensem_model_parser = subparsers.add_parser('other-ensem')
    other_ensem_model_parser.add_argument('--voting',action='store_true')
    other_ensem_model_parser.add_argument('--xg',action='store_true')

   
    args=parser.parse_args()
    
    if args.subparser_name=='train-models':
        train_models(args)
    
    elif args.subparser_name=='other-ensem':
        other_ensem(args)


