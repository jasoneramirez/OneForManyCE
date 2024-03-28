###one-for-many counterfactuals

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from time import time
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer



#from joblib import load #to import a pretrained nn
from gurobi_ml import add_predictor_constr

from column_gen_counterf_gurobipy_heuristics import*

import torch.nn as nn
import torch
import torchvision
from skorch import NeuralNetClassifier


#compas 5 layers 20 width 

class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, num_layers, activation=nn.ReLU(), output_activation=nn.Softmax(dim=1)):
        super(CustomNeuralNetwork, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(activation)
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(output_activation)
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

def neural_network(x_train, y_train, input_size, output_size, hidden_sizes, num_layers, max_epochs=500, lr=0.1):
    if isinstance(x_train, pd.DataFrame):
        x_train = x_train.to_numpy()
        x_train = torch.tensor(x_train, dtype=torch.float32)
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
        y_train = torch.tensor(y_train, dtype=torch.long)

    nn_model = CustomNeuralNetwork(input_size, output_size, hidden_sizes, num_layers)
    
    clf = NeuralNetClassifier(
        nn_model,
        max_epochs=max_epochs,
        lr=lr,
        iterator_train__shuffle=True,
        criterion=torch.nn.CrossEntropyLoss()
    )

    clf.fit(X=x_train, y=y_train)
    
    nn_regression = torch.nn.Sequential(*nn_model.layers[:-1])

    print(f"Training score: {clf.score(x_train, y_train):.4}")

    return clf, nn_regression, nn_model.model


#Logistic regression
def modelo(x_train,y_train,tipo):
    #Logistic Regresion
    if tipo=='LR':
        model = LogisticRegression(solver='liblinear', random_state=0,C=10.0) #C regularizacion
        model.fit(x_train,y_train)

        y_pred=model.predict(x_train)
        print('Training score: '+str(accuracy_score(y_train,y_pred)))
        w=model.coef_
        b=model.intercept_    
        
    return model, w,b


#Data
def datos(dataset):
    if dataset=='compas':
        compas=pd.read_csv("compas_processed.csv",sep=";")
        compas['TwoYearRecid']=compas['TwoYearRecid'].apply(lambda x: 1 if x==0 else 0) #positive class 1 is no recid
        compas.loc[compas['PriorsCount'] > 3, 'PriorsCount'] = 'Mas3'
        compas.loc[compas['PriorsCount'] == 0, 'PriorsCount'] = 'Ninguna'
        compas.loc[compas['PriorsCount'] == 1, 'PriorsCount'] = 'Una'
        compas.loc[(compas['PriorsCount'] == 2) | (compas['PriorsCount']==3), 'PriorsCount'] = 'Pocas'
        x_compas=compas.drop(columns=['TwoYearRecid'])
        y=compas['TwoYearRecid']

        to_onehot=['Race','AgeGroup','PriorsCount']
        df = x_compas
        df = df.dropna()
        transformer = make_column_transformer(
            (OneHotEncoder(), to_onehot),
            remainder='passthrough')
        transformed = transformer.fit_transform(df)
        transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
        variables_onehot=[c for c in transformed_df.columns if 'onehot' in c]
        for col in variables_onehot:
            transformed_df[col]=transformed_df[col].astype('category')
        var_related={to_onehot[i]: [variables_onehot.index(char) for char in variables_onehot if to_onehot[i] in char] for i in range(len(to_onehot))}

        compas2=transformed_df
        
        
        bin=['Sex','ChargeDegree']
        for col in bin:
            var_related[col]=[compas2.columns.get_loc('remainder__'+col)]

        compas2.rename(columns=lambda x: x.replace('remainder__', ''), inplace=True)

        
        x=compas2

        T_original=x_compas.shape[1]

        T_corr={}
        for h in range(T_original):
            T_corr[h]=list(var_related.values())[h]

        T_c=[element for element, list in T_corr.items() if len(list)>1]

        sets_T=[T_original,T_corr,T_c]

        name_features=['Race','AgeGroup','PriorsCount','Sex','ChargeDegree']

    elif dataset=="Students":
        students=pd.read_csv("Students-Performance-MAT.csv")
        students['Class']=students['Class'].apply(lambda x: 1 if x==1 else 0)
      
        #binary variables to 0-1
        students['school']=students['school'].apply(lambda x: 1 if x=='MS' else 0)
        students['sex']=students['sex'].apply(lambda x: 1 if x=='F' else 0)
        students['address']=students['address'].apply(lambda x: 1 if x=='U' else 0)
        students['famsize']=students['famsize'].apply(lambda x: 1 if x=='GT3' else 0)
        students['Pstatus']=students['Pstatus'].apply(lambda x: 1 if x=='A' else 0)
        students['schoolsup']=students['schoolsup'].apply(lambda x: 1 if 'yes' else 0)
        students['famsup']=students['famsup'].apply(lambda x: 1 if 'yes' else 0)
        students['paid']=students['paid'].apply(lambda x: 1 if 'yes' else 0)
        students['activities']=students['activities'].apply(lambda x: 1 if 'yes' else 0)
        students['nursery']=students['nursery'].apply(lambda x: 1 if 'yes' else 0)
        students['higher']=students['higher'].apply(lambda x: 1 if 'yes' else 0)
        students['internet']=students['internet'].apply(lambda x: 1 if 'yes' else 0)
        students['romantic']=students['romantic'].apply(lambda x: 1 if 'yes' else 0)
        
        #numeric to categorical (either binary or more)
        students.loc[students['age'] >=18, 'age'] = 'More18'
        students.loc[(students['age'] ==15) | (students['age'] ==16), 'age'] = '15-16'
        students.loc[(students['age'] ==16) | (students['age'] ==17) | (students['age'] ==18), 'age'] = '16-18'

        students.loc[students['failures'] >0, 'failures'] = 1
        students.loc[students['failures'] ==0, 'failures'] = 0

    

        students.loc[(students['absences'] > 0) & (students['absences'] <=10), 'absences'] = '1-10'
        students.loc[students['absences'] ==0, 'absences'] = 'None'
        students['absences'] = np.where(students['absences'].apply(lambda x: pd.notna(x) and isinstance(x, (int, float))), 'More10', students['absences'])

        to_onehot=["age","Medu","Fedu","Mjob","Fjob","reason","guardian","traveltime","studytime","famrel","freetime","goout","Dalc","Walc","health","absences"]

        df = students
        df = df.dropna()
        transformer = make_column_transformer(
            (OneHotEncoder(), to_onehot),
            remainder='passthrough')
        transformed = transformer.fit_transform(df)
        transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
        variables_onehot=[c for c in transformed_df.columns if 'onehot' in c]
        for col in variables_onehot:
            transformed_df[col]=transformed_df[col].astype('category')
        

        var_related={to_onehot[i]: [variables_onehot.index(char) for char in variables_onehot if to_onehot[i] in char] for i in range(len(to_onehot))}
        students2=transformed_df
        
        bin=['school','sex','address','famsize','Pstatus','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
        for col in bin:
            var_related[col]=[students2.columns.get_loc('remainder__'+col)]

        students2.rename(columns=lambda x: x.replace('remainder__', ''), inplace=True)
        x=students2.drop(columns=['Class'])
        y=students2['Class']

        T_original=students.shape[1]-1

        T_corr={}
        for h in range(T_original):
            T_corr[h]=list(var_related.values())[h]

        T_c=[element for element, list in T_corr.items() if len(list)>1]

        sets_T=[T_original,T_corr,T_c]

        name_features=list(var_related.keys())


    elif dataset=='german':
        columns_german=['status_account','duration','credit_history','purpose','amount','savings',
                        'employment','installment_rate','status_sex','guarantors','residence_since',
                        'property','age','installment_plans','housing','number_credits','job',
                        'maintenance_obligors','tel','foreign','target']
        german=pd.read_csv('german.data', sep=" ",header=None,names=columns_german)
        german['target']=german['target'].apply(lambda x: 1 if x==1 else 0)

        #numeric to binary
        german.loc[german['duration'] <=18, 'duration'] = 0
        german.loc[german['duration'] >18, 'duration'] = 1

        german.loc[german['amount']<=2319.5, 'amount']=0
        german.loc[german['amount']>2319.5,'amount']=1

        german.loc[german['installment_rate']<=3, 'installment_rate']=0
        german.loc[german['installment_rate']>3,'installment_rate']=1

        german.loc[german['residence_since']<=3, 'residence_since']=0
        german.loc[german['residence_since']>3,'residence_since']=1

        german.loc[german['age']<=33, 'age']=0
        german.loc[german['age']>33,'age']=1

        german.loc[german['number_credits']<=1, 'number_credits']=0
        german.loc[german['number_credits']>1,'number_credits']=1

        #this was already binary
        german.loc[german['maintenance_obligors']<=1,'maintenance_obligors']=0
        german.loc[german['maintenance_obligors']>1,'maintenance_obligors']=1

        #binary to 0,1
        german['tel']=german['tel'].apply(lambda x: 1 if x=='A192' else 0)
        german['foreign']=german['foreign'].apply(lambda x: 1 if x=='A201' else 0)

        to_onehot=['status_account','credit_history','purpose','savings','employment','status_sex','guarantors', 
                   'property','installment_plans','housing','job']
        
        df = german
        df = df.dropna()
        transformer = make_column_transformer(
            (OneHotEncoder(), to_onehot),
            remainder='passthrough')
        transformed = transformer.fit_transform(df)
        transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
        variables_onehot=[c for c in transformed_df.columns if 'onehot' in c]
        for col in variables_onehot:
            transformed_df[col]=transformed_df[col].astype('category')
        

        var_related={to_onehot[i]: [variables_onehot.index(char) for char in variables_onehot if to_onehot[i] in char] for i in range(len(to_onehot))}
        german2=transformed_df

        bin=['duration','amount','installment_rate','residence_since','age','number_credits','maintenance_obligors','tel','foreign']
        for col in bin:
            var_related[col]=[german2.columns.get_loc('remainder__'+col)]

        german2.rename(columns=lambda x: x.replace('remainder__', ''), inplace=True)
        x=german2.drop(columns=['target'])
        y=german2['target']

        T_original=german.shape[1]-1

        T_corr={}
        for h in range(T_original):
            T_corr[h]=list(var_related.values())[h]

        T_c=[element for element, list in T_corr.items() if len(list)>1]

        sets_T=[T_original,T_corr,T_c]

        name_features=list(var_related.keys())
        

                        




    return x,y, sets_T, name_features


x,y,sets_T,name_features=datos("Students")
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.5,random_state=0)

#LR
#model, w,b=modelo(x_train,y_train,"LR")
#print(accuracy_score(y_test,model.predict(x_test))) #0.7333 #0.587
#y_pred_test_LR=model.predict(x_test)
#x0=x[y_pred_total==0]



#NN
clf, nn_regression, nn_model = neural_network(x_train, y_train, input_size=x.shape[1], output_size=2, hidden_sizes=[7,7], num_layers=2)






clas_model={}
clas_model[0]='NN' #or 'LR'
clas_model[1]=nn_regression #or model_clas with w and b


def init_sol_milp(X, sets_T, Tmax_, clas_model):
    
        T_original=sets_T[0]
        T_corr=sets_T[1]
        T_c=sets_T[2]

        X_=X
        n_samples=X_.shape[0]
        n_features=X_.shape[1]
        n_counterf=n_samples


        m=gp.Model('initsol')

        x_l={}
        for k in range(n_samples):
            x_l[k]=m.addMVar(n_features, vtype=GRB.BINARY, name="counterf[%d]"%k)
            #x_l[k]={}
            #for f in range(self.n_features):
            #    x_l[k][f]=m.addVar(vtype=GRB.BINARY,name="counterf[%d][%d]"%(k,f))

        y_clas={}
        for k in range(n_samples):
            y_clas[k]=m.addMVar(2, lb=-gp.GRB.INFINITY, name="y[%d]"%k)

        xi={}
        for k in range(n_samples):
            xi[k]={}
            for f in range(n_features):
                xi[k][f]=m.addVar(vtype=GRB.BINARY,name="abs[%d][%d]"%(k,f))

        feat ={}
        for k in range(n_samples):
            feat[k]={}
            for j in range(T_original):
                feat[k][j]=m.addVar(vtype=GRB.BINARY,name="feat[%d][%d]"%(k,j))

        for k in range(n_samples):
            for f in range(n_features):
                m.addConstr(-xi[k][f]<=X_.iloc[k,f]-x_l[k][f])
                m.addConstr(X_.iloc[k,f]-x_l[k][f]<=xi[k][f])

        
        #Add constraint to see if original feature is used
        for k in range(n_samples):
            for l in range(T_original): 
                m.addConstr(len(T_corr[l])*feat[k][l]>= gp.quicksum(xi[k][j] for j in T_corr[l]),name="SparseOriginalFeature")


        #Add max features constraint
        for k in range(n_samples):
            m.addConstr(gp.quicksum(feat[k][h] for h in range(T_original))<=Tmax_, name="MaxFeaturesCons")

        #Add constraint ensuring the one-hot encoding

        for k in range(n_samples):
            for l in T_c:
                m.addConstr(gp.quicksum(x_l[k][h] for h in T_corr[l])==1, name="Onehot")
     
       

        #ensure is a positive sample
        if clas_model[0]=='LR':
            w=clas_model[1].coef_
            b=clas_model[1].intercept_   
            for k in range(n_samples):
                m.addConstr(gp.quicksum(w[0][f]*x_l[k][f] for f in range(n_features))+b[0]>=0)

        elif clas_model[0]=='NN':
            nn_regression=clas_model[1]
            for k in range(n_samples):
                add_predictor_constr(m, nn_regression, x_l[k], y_clas[k])
                m.addConstr(y_clas[k][1]-y_clas[k][0]>=1e-3)

        m.optimize()

        counterf_init={}
        for k in range(n_samples):
            counterf_init[k]=[]
            for f in range(n_features):
                counterf_init[k].append(x_l[k][f].getAttr(GRB.Attr.X))

        list_counterf=list(counterf_init.values())
        
        unique_counterf = [element for index, element in enumerate(list_counterf) if element not in list_counterf[:index]]

        #features_used_init={}
        #for k in range(n_samples):
        #    features_used_init[k]=[]
        #    for f in range(T_original):
        #        if feat[k][f].getAttr(GRB.Attr.X)==1:
        #            features_used_init[k].append(f)



        #features_used_init_def={}
        #for i, item in enumerate(list_counterf):
        #    for j,element in enumerate(unique_counterf):
        #        if item==element:
        #            features_used_init_def[j]=features_used_init[i]

        features_init={}
        for k in range(n_samples):
            features_init[k]={}
            for f in range(n_features):
                features_init[k][f]=xi[k][f].getAttr(GRB.Attr.X)

        features_init_def={}
        for i, item in enumerate(list_counterf):
            for j, element in enumerate(unique_counterf):
                if item==element:
                    features_init_def[j]={}
                    for f in range(n_features):
                        features_init_def[j][f]=features_init[i][f]


        A_init = np.zeros((len(list_counterf), len(unique_counterf)), dtype=int)

        for i, item in enumerate(list_counterf):
            for j, element in enumerate(unique_counterf):
                if item == element:
                    A_init[i, j] = 1

        sol_init=[]
        sol_init.append(unique_counterf)
        sol_init.append(A_init)
        sol_init.append(features_init_def)

        return sol_init



def global_cf_sparse_cluster(clas_model,x0,sets_T,phinu,F,K_max,sol_init):

    features=list(x0.columns)
    n_features=x0.shape[1]
    n_samples=x0.shape[0]
    T_original=sets_T[0]
    T_corr=sets_T[1]
    T_c=sets_T[2]

    ce_init=sol_init[0]
    A_init=sol_init[1].transpose()
    xi_init=sol_init[2]

    initial_solution = {
    "x_l": {k: ce_init[k] for k in range(len(ce_init))},
    "xi": {(k, f): xi_init[k][f] for k in range(len(xi_init)) for f in range(n_features)},
    "p": {(k, i): A_init[k][i] for k in range(A_init.shape[0]) for i in range(n_samples)},
    "y_sel": {k: 1 for k in range(len(ce_init))}}


    m=gp.Model("global_cf")
    
    #maximum number of explanations
    K=K_max #poner un numero tan grande es un desproposito
    #Decision variables

    #x_l={}
    #for k in range(K):
    #    x_l[k]={}
    #    for f in range(n_features):
    #        x_l[k][f]=m.addVar(vtype=GRB.BINARY,name='counterf[%d][%d]'%(k,f))

    x_l={}
    for k in range(K):
        x_l[k]=m.addMVar(n_features, vtype=GRB.BINARY, name="counterf[%d]"%k)

    y={}
    for k in range(K):
        y[k]=m.addMVar(2, lb=-gp.GRB.INFINITY, name="y[%d]"%k)



    xi={}
    for k in range(K):
        for f in range(n_features):
            xi[k,f]=m.addVar(vtype=GRB.BINARY, name='xi[%d][%d]'%(k,f))

    xi_og={}
    for k in range(K):
        for f in range(T_original):
            xi_og[k,f]=m.addVar(vtype=GRB.BINARY, name='xi_og[%d][%d]'%(k,f))
    
    #for the absolute value
    xi_aux={}
    for k in range(K):
        for i in range(n_samples):
            for f in range(n_features):
                xi_aux[k,i,f]=m.addVar(vtype=GRB.BINARY, name='xi_aux[%d][%d][%d]'%(k,i,f))

    #for the product
    xi_aux_prod={}
    for k in range(K):
        for i in range(n_samples):
            for f in range(n_features):
                xi_aux_prod[k,i,f]=m.addVar(vtype=GRB.BINARY, name="aux[%d][%d][%d]"%(k,i,f))
    #assignment
    p={}
    for k in range(K):
        for i in range(n_samples):
            p[k,i]=m.addVar(vtype=GRB.BINARY, name="assignment[%d][%d]"%(k,i))
    
    #selection of counterf
    y_sel={}
    for k in range(K):
        y_sel[k]=m.addVar(vtype=GRB.BINARY,name="used_expl[%d]"%k)



    #Objective

    

    m.setObjective(gp.quicksum(y_sel[k] for k in range(K)), GRB.MINIMIZE)

    #constraints f>=0

    #pred_constr.print_stats()
    
    if clas_model[0]=='LR':
        w=clas_model[1].coef_
        b=clas_model[1].intercept_   
        for k in range(K):
            m.addConstr(gp.quicksum(w[0][f]*x_l[k][f] for f in range(n_features))+b[0]>=0)

    elif clas_model[0]=='NN':
        nn_regression=clas_model[1]
        for k in range(K):
            add_predictor_constr(m, nn_regression, x_l[k], y[k])
            m.addConstr(y[k][1]-y[k][0]>=1e-3)

   
    #the absolute value
    for k in range(K):
        for i in range(n_samples):
            for f in range(n_features):
                m.addConstr(-xi_aux[k,i,f]<=x0.iloc[i,f]-x_l[k][f])
                m.addConstr(x0.iloc[i,f]-x_l[k][f]<=xi_aux[k,i,f])
           
    for k in range(K):
        for i in range(n_samples):
            for f in range(n_features):
                m.addConstr(xi[k,f]>=xi_aux_prod[k,i,f])

    for k in range(K):
        for i in range(n_samples):
            for f in range(n_features):
                m.addConstr(xi_aux_prod[k,i,f]<=xi_aux[k,i,f])
                m.addConstr(xi_aux_prod[k,i,f]<=p[k,i])
                m.addConstr(xi_aux_prod[k,i,f]>=xi_aux[k,i,f]+p[k,i]-1)

    #see if original feature used
    for k in range(K):
        for l in range(T_original):
            m.addConstr(len(T_corr[l])*xi_og[k,l]>=gp.quicksum(xi[k,f] for f in T_corr[l]), name="SparseOg[%d][%d]"%(k,l))
           
    #max feature
    for k in range(K):
        m.addConstr(gp.quicksum(xi_og[k,f] for f in range(T_original))<=F)

    #one-hot
    for k in range(K):
        for l in T_c:
            m.addConstr(gp.quicksum(x_l[k][f] for f in T_corr[l])==1, name="Onehot[%d]"%k)

    for i in range(n_samples):
        m.addConstr(gp.quicksum(p[k,i] for k in range(K))>=1)

    for k in range(K):
        m.addConstr(gp.quicksum(p[k,i] for i in range(n_samples))<=n_samples*y_sel[k])
    
    m.setParam('TimeLimit', 4200)
    #m.write('modelo_nn.lp')


    # Apply initial solutions to variables
    for var_type, var_dict in {"x_l": x_l, "xi": xi, "p": p, "y_sel":y_sel}.items():
        for indices, initial_values in initial_solution[var_type].items():
            var = var_dict[indices]
            if isinstance(var, gp.Var):  # Check if var is a single variable
                var.setAttr(GRB.Attr.Start, initial_values)
            else:  # Iterate over the MVar (variable list)
                for j, val in enumerate(initial_values):
                    var[j].setAttr(GRB.Attr.Start, val)
    


    time_init=time()
    m.optimize()
    
    #m.optimize(callback=stop_subproblem)
    time_solve=time()-time_init

    if m.status == GRB.OPTIMAL:
        print("Optimal solution found:")
    for var_type, var_dict in {"x_l": x_l, "xi": xi, "p": p, "y_sel": y_sel}.items():
        for indices, var in var_dict.items():
            print(f"{var_type}{indices}: Initial={var.Start}, Final={var.x}")
    else:
        print("No solution found.")


    #explanation used
    expl=[]
    for k in range(K):
        if y_sel[k].getAttr(GRB.Attr.X)==1:
            expl.append(k)

    counterf={}
    for k in expl:
        counterf[k]={}
        for f in range(n_features):
            counterf[k][f]=x_l[k][f].getAttr(GRB.Attr.X)

    cfs=pd.DataFrame.from_dict(counterf, orient='index',dtype=float)

    assignment={}
    for k in expl:
        assignment[k]=[]
        for i in range(n_samples):
            if p[k,i].getAttr(GRB.Attr.X)==1:
                 assignment[k].append(i)

    cluster={}
    for k in expl:
        cluster[k]=[]
        for i in range(n_samples):
            if p[k,i].getAttr(GRB.Attr.X)==1:
                cluster[k].append(x0.iloc[i])


    #for i in range(n_ind):
    #    change={}
    #    for f in range(n_categories+n_continuos):
    #        change[feature_names[f]]=x_l[feature_names[f]].getAttr(GRB.Attr.X)-x0[i][f]
    #    changes.append(change)

    #chang=pd.DataFrame(changes)

    xi_f={}
    for k in expl:
        xi_f[k]=[]
        for f in range(n_features):
            if xi[k,f].getAttr(GRB.Attr.X)==1:
                xi_f[k].append(features[f])

    
    num=len(expl)

    return cfs, cluster, assignment, xi_f, num, time_solve


x0=x_test
data_milp={}
for T in [20]:
    data_milp[T]={}
    for n_ind in [100]:
        data_milp[T][n_ind]={}
        x0p=x0[0:n_ind]
        phinu=0
        sol_init=init_sol_milp(x0p,sets_T,T,clas_model)
        K_max=len(sol_init[0])
        counterf, cluster, assignment, xi_f, num_counterf,time_solve=global_cf_sparse_cluster(clas_model,x0p,sets_T,phinu,T,K_max,sol_init)
        data_milp[T][n_ind]['n_cfe']=num_counterf
        data_milp[T][n_ind]['t_milp']=time_solve
    df_milp=pd.DataFrame(data_milp[T]).transpose()
    df_milp.astype(float)
    df_milp.to_csv('students_milp_Tmax_'+str(T)+'_NN_extra_f.csv',sep=";",header=True,decimal=",")



#x0=x0_final

data={}
for T in [20]:
    data[T]={}
    for n_ind in [20]:
        data[T][n_ind]={}
        x0p=x0[0:n_ind]
        #subproblem_params={'timelimit':800,'threshold':1.1}
        subproblem_params={'timelimit':800}
        pool=True
        pool_obj1=False
        auxiliar_milp=True
        master_params={'timelimit':800}
        globalcounterf=GlobalCounterfColGen(Tmax=T,counterfactual_instance=True,rmp_is_ip=True,max_iterations=2000,time_limit=3600,subproblem_params=subproblem_params,rmp_solver_params=master_params,pool=pool,pool_obj1=pool_obj1,auxiliar_milp=auxiliar_milp)
        cf,assign,f_sol=globalcounterf.fit(x0p,sets_T,clas_model)
        data[T][n_ind]['n_cfe']=len(cf)
        data[T][n_ind]['t_sp']=globalcounterf.time_spent_sp_[0][0] 
        data[T][n_ind]['t_add_col']=globalcounterf.time_add_col_
        data[T][n_ind]['t_master']=globalcounterf.time_spent_master_ 
        data[T][n_ind]['t_final_mip']=globalcounterf.time_spent_master_ip_
        data[T][n_ind]['gap']=globalcounterf.gap
        data[T][n_ind]['total_iter']=globalcounterf.iter
        data[T][n_ind]['n_columns_added']=globalcounterf.num_col_added_sp_[0][0]
        data[T][n_ind]['features']=[[name_features[i] for i in indices_list] for indices_list in list(f_sol.values())]
        data[T][n_ind]['values_cfes']=cf
        data[T][n_ind]['assignments']=assign
        data[T][n_ind]['pool']=pool
        data[T][n_ind]['pool_obj1']=pool_obj1
        data[T][n_ind]['auxiliar_milp']=auxiliar_milp

    df=pd.DataFrame(data[T]).transpose()
    columns_to_cast =df.columns.difference(['features','values_cfes','assignments','pool','pool_obj1','auxiliar_milp'])
    df[columns_to_cast] = df[columns_to_cast].astype(float)
    df.to_csv('students_cg_Tmax_'+str(T)+'_LR_extra2.csv',sep=";",header=True,decimal=",")
    #df.to_pickle('students_cg_Tmax_'+str(T)+'_LR_extra2.pkl')

 