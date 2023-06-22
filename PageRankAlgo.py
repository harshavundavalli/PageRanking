#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the required libraries
from numpy import linalg as la
import numpy as np


# In[2]:


class Graph(object):
    
    """
    This class is to generate the adjancency matrix\n
    """

    # Initialize the matrix
    def __init__(self,size):
        self.adjMatrix1 = np.zeros((size,size))
        self.adjMatrix2 = np.zeros((size,size))
        self.size = size

    # Add edges
    def add_edge(self, v1, v2):
        self.adjMatrix1[v1][v2] = 1
        self.adjMatrix2[v1][v2] = 1


# In[3]:


file1 = open('input.txt', 'r')
Lines = file1.readlines()

nodes = int(Lines[0])
connections = int(Lines[1])

g = Graph(nodes)

for i in range(2,connections+2):
    x,y = Lines[i].split(',')
    x,y = int(x),int(y)
    g.add_edge(x-1,y-1)
mat = g.adjMatrix1
mat = mat.transpose()
mat


# In[4]:


mat1 = mat


# In[5]:


mat = np.array(mat,dtype=np.float32)


# In[6]:


mat


# In[7]:


for i in range(len(mat[0])):
    mat[:,i] = mat[:,i]/sum(mat[:,i])


# In[8]:


mat


# In[9]:


alpha = 0.1
matTr = (1-alpha)*mat 
#+ (alpha/len(mat))*np.ones(mat.shape)
A = matTr==0
A = np.array(A,dtype = np.float32)
for i in range(len(A[0])):
    A[:,i] = A[:,i]/sum(A[:,i])
matTr = matTr + alpha*A


# In[10]:


matTr


# In[11]:


lamdas, evs = la.eig(mat)


# In[12]:


def findIndex(lams):
    """ This gives us the index of principle eigen value """
    ind = np.argmax(lams);
    #li = list(lams)
    #for i in range(len(li)):
     #   if(np.round(li[i], 4) == 1.0):
      #      ind = i
       #     break
    #print(li[ind])
    return ind


# ## Without teleportation using algebra

# In[13]:


ind = findIndex(lamdas)
lpev = evs[:,ind]
lpev = np.reshape(lpev, (lpev.shape[0],1))
lpev = lpev/(sum(sum(lpev)))
lpev


# In[14]:


sum(sum(lpev))


# In[15]:


p = []
for i in range(len(lpev)):
    p.append([1/(len(lpev))])
p = np.array(p)


# In[16]:


p


# ## Without teleportation using Iterative

# In[17]:


while True:
    tp = np.dot(mat,p)
    if(np.all(abs(tp-p) < 0.00001)):
        break
    p = tp


# In[18]:


p


# In[19]:


lamdasTr, evsTr = la.eig(matTr)


# ## With teleportation using algebra

# In[20]:


ind = findIndex(lamdasTr)
lpev = evsTr[:,ind]
lpev = np.reshape(lpev, (lpev.shape[0],1))
lpev = lpev/(sum(sum(lpev)))
lpev


# In[21]:


p = []
for i in range(len(lpev)):
    p.append([1/(len(lpev))])
p = np.array(p)


# In[22]:


p


# ## With teleportation using Iterative

# In[23]:


while True:
    tp = np.dot(matTr,p)
    if(np.all(abs(tp-p) < 0.00001)):
        break
    p = tp


# In[24]:


p


# In[25]:


def AlgWithoutTele(mat):
    """
    Adjacency matrix of the given graph is taken as parameter for this funcion\n
    This Function generates page rank vectors without teleportation in both iterative method and the algebraic method.\n
    In algebraic method the principal left eigen vector of the gives us the page rank vectors for the given graph\n
    In iterative method a vector is taken initially and it is multiplied with the adjacency matrix until there is no much change in the vector obtained on multiplication\n
    """
    mat = np.array(mat,dtype=np.float32)
    for i in range(len(mat[0])):
        mat[:,i] = mat[:,i]/sum(mat[:,i])
    lamdas, evs = la.eig(mat)
    ind = findIndex(lamdas)
    lpev = evs[:,ind]
    lpev = np.reshape(lpev, (lpev.shape[0],1))
    lpev = lpev/(sum(sum(lpev)))
    print(lpev)
    print()
    p = []
    for i in range(len(lpev)):
        p.append([1/(len(lpev))])
    p = np.array(p)
    while True:
        tp = np.dot(mat,p)
        if(np.all(abs(tp-p) < 0.00001)):
            break
        p = tp
    print(p)


# In[26]:


def AlgWithTele(mat):
    """
    Adjacency Matrix of the given graph is taken as parameter for this funcion\n
    This Function generates page rank vectors with teleportation in both iterative method and the algebraic method.\n
    """
    mat = np.array(mat,dtype=np.float32)
    for i in range(len(mat[0])):
        mat[:,i] = mat[:,i]/sum(mat[:,i])
    alpha = 0.1
    matTr = (1-alpha)*mat 
    #+ (alpha/len(mat))*np.ones(mat.shape)
    A = matTr==0
    A = np.array(A,dtype = np.float32)
    for i in range(len(A[0])):
        A[:,i] = A[:,i]/sum(A[:,i])
    mat = matTr + alpha*A
    
    #mat = np.array(mat,dtype=np.float32)
    #for i in range(len(mat[0])):
     #   mat[:,i] = mat[:,i]/sum(mat[:,i])
    lamdas, evs = la.eig(mat)
    
    ind = findIndex(lamdas)
    lpev = evs[:,ind]
    lpev = np.reshape(lpev, (lpev.shape[0],1))
    lpev = lpev/(sum(sum(lpev)))
    print(lpev)
    print()
    p = []

    for i in range(len(lpev)):
        p.append([1/(len(lpev))])
    p = np.array(p)
    while True:
        tp = np.dot(mat,p)
        if(np.all(abs(tp-p) < 0.00001)):
            break
        p = tp
    print(p)
   
    


# In[27]:


AlgWithoutTele(mat1)


# In[28]:


AlgWithTele(mat1)


# In[30]:


from networkx.generators.random_graphs import erdos_renyi_graph
import networkx as nx
import time
dict1 = {}

for i in range(4,60):
    n=i
    for j in range(6):
        p = 0.5
        graph = erdos_renyi_graph(n, p,directed = True)
        Adj = nx.adjacency_matrix(graph)
        Adj = Adj.todense()
        Adj = np.array(Adj,dtype=np.float32)
        Adj = Adj.transpose()
        #mat = np.array(mat,dtype=np.float32)
        flag = True
        for k in range(len(Adj[0])):
            val = sum(Adj[:,k])
            if val==0 :
                flag = False
                break;
            Adj[:,k] = Adj[:,k]/val
        if not flag :
            continue
        start = time.time() 
        AlgWithoutTele(Adj)
        AlgWithTele(Adj)
        end = time.time()
        time_diff = end - start
        dict1[len(graph.edges)] = time_diff
            
            
        
        
        
        
        
        
        
       


# In[31]:


import collections
dict1 = collections.OrderedDict(sorted(dict1.items()))
dict1


# In[32]:


import matplotlib.pyplot as plt

x = dict1.keys()
y = dict1.values()
 
# Plotting the Graph
plt.plot(x, y)
plt.title("Plot of Runtime vs Number of edges")
plt.xlabel("Number Of Edges")
plt.ylabel("Runtime")
plt.show()


# In[ ]:




