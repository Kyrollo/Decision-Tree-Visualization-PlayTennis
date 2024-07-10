import numpy as np
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

data=pd.read_csv("PlayTennis.csv")
print("data: ")
print(data)
print("\ndata statistics: ")
print(data.describe())
print("")

X=data[['Outlook','Temperature','Humidity','Wind']]
X=X.values
Y=data['Play Tennis'].values
Y=Y.reshape([-1,1])

def Entropy(p,n):
    sum=p+n
    if n==0 and p==0:
        return 0
    if n == 0:
        return int(-(p / sum) * np.log2(p / sum))
    elif p == 0:
        return int(-(n / sum) * np.log2(n / sum))
    else:
        return np.sum(-(p/sum)*np.log2(p/sum)-(n/sum)*np.log2(n/sum))
    
def Avg_entropy(propapility,entropy):
    return np.sum(propapility*entropy)

def Gain_information(entorpy,avg_entropy):
    return entorpy-avg_entropy

def calculate_gain_information(X,master_entropy,label,condition):
    gain_information = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x= np.array(X[:, i]).reshape(-1, 1)
        unique_label=np.unique(x)
        prop=np.zeros(len(unique_label))
        entropy=np.zeros(len(unique_label))
        for j in range(len(unique_label)):
            p=np.sum(("Yes"==Y) & (unique_label[j]==x) & (condition))
            n = np.sum(("No"==Y) & (unique_label[j]==x) & (condition))
            entropy[j]=Entropy(p,n)
            prop[j]=(p+n)/np.sum(condition)

        avg_entropy=Avg_entropy(prop,entropy)
        gain_info=Gain_information(master_entropy,avg_entropy)
        gain_information[i]=gain_info

    print("gain_information: ",gain_information)
    greatest_gain_info_value=np.max(gain_information)
    greatest_gain_info_index=np.argmax(gain_information)

    chosen_label=label[greatest_gain_info_index]
    print("chosen label: ",chosen_label)
    print("")
    return chosen_label

def calculate_entropy(unique_label,index,condition=True):
    Decision_tree_array=[]
    entropy = np.zeros(len(unique_label))
    decision1=[]
    decision2=[]
    for j in range(len(unique_label)):
        x = np.array(X[:, index]).reshape(-1, 1)
        p = np.sum(("Yes" == Y) & (unique_label[j] == x) & condition)
        n = np.sum(("No" == Y) & (unique_label[j] == x) & condition)
        entropy[j] = Entropy(p, n)
        if p == 0:
            decision2+=["No"]
            decision1+=[unique_label[j]]
        if n == 0:
            decision2+=["Yes"]
            decision1+=[unique_label[j]]
    return entropy,decision1,decision2

print("build the tree: ")
p=np.sum("Yes"==Y)
n=Y.shape[0]-p
play_tennis_entropy=Entropy(p,n)
boolean_array = np.full((14,1), True)
root=np.unique(X[:,0])
golden_label=np.array(['Outlook','Temperature','Humidity','Wind'])
feature_order=[]
tree=[]
nodes=[]
label=np.array(['Outlook','Temperature','Humidity','Wind'])
chosen_label1 = calculate_gain_information(X, play_tennis_entropy, label, boolean_array)
index=np.where(golden_label == chosen_label1)[0]
entropy2, decision1,decision2 = calculate_entropy(root, index)
mask = label != chosen_label1
label = label[mask]
delete_index = [index]
feature_order+=[chosen_label1]
dot = Digraph()
nodes.append(chosen_label1)

def build_tree(chosen_label1,index,entropy2,label,root):
    feature_order=[]
    delete_index = [index]
    feature_order += [chosen_label1]
    for j in range((len(entropy2)-1), -1, -1):
        if(entropy2[j]!=0):
            array_without_column = np.delete(X, delete_index, axis=1)
            chosen_label2 = calculate_gain_information(array_without_column, entropy2[j], label,
                                          (X[:, index] == root[j]).reshape(-1, 1))
            index2 = np.where(golden_label == chosen_label2)[0]
            mask = label != chosen_label2
            label = label[mask]
            feature = np.unique(X[:, index2])
            entropy3, decision3,decision4 = calculate_entropy(feature, index2,
                                                            (X[:, index] == root[j]).reshape(-1, 1))
            if np.all(entropy3)!=0:
                build_tree(chosen_label2,index2,entropy3,label,feature[entropy3!=0])
            delete_index+=[index2]
            feature_order += [chosen_label2]
            nodes.append(chosen_label2)
            for item in decision4:
                nodes.append(item)
            dot.edge(chosen_label1, chosen_label2, color='red', label=root[j], style='dashed')
            for i in range(len(decision3)):
                dot.edge(chosen_label2, decision4[i], color='red', label=decision3[i], style='dashed')
    for i in range(len(decision1)):
        dot.edge(chosen_label1, decision2[i], color='red', label=decision1[i], style='dashed')

    print("features used by order: ",feature_order)
    
build_tree(chosen_label1,index,entropy2,label,root)
for item in nodes:
    dot.node(item)
dot.render('graph', format='png', cleanup=True)
dot.view()
