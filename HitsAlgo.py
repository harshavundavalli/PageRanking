#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import matplotlib.pyplot as plt
import time
import collections
import scipy as sp


# In[2]:


"""Taking the given graph as input into G"""
G = nx.read_gpickle("web_graph.gpickle")
G


# In[3]:


G.nodes[0]['page_content']


# In[4]:



def draw(G):
    """ This function is to draw the graph"""
    pos = {i: G.nodes[i]['pos'] for i in range(len(G.nodes))}
    nx.draw(G, pos,with_labels=True)


# In[5]:


len(G.nodes)
"""Edges present in the graph"""
G.edges


# In[6]:

""" Adjacency Matrix for the given graph G"""
Adj = nx.adjacency_matrix(G)
Adj


# In[7]:


Adj = Adj.todense()


# In[8]:


Adj


# In[9]:


Adj = np.array(Adj)
Adj


# In[10]:



def get_words(st):
    """This is the preprocessing step which gives the words in the corpus removing corpus\n
    Stopwords removal and lemmatization is done on the data\n
    :param
        data in the document/query
    :returns
        List of words after performing the preprocessing on data
    """

    st = st.lower()
    st = ''.join(ch for ch in st if ch.isalnum() or ch==' ')
    words  = st.split()
    stpwrds = set(stopwords.words('english'))
    newwords = set()
    for word in words:
        word = lemmatizer.lemmatize(word)
        if word not in stpwrds and word != ' ' and word !='' :
            newwords.add(word)

    newwords = list(newwords)
    if len(newwords) == 0:
        print("True")
    return newwords
    


# In[11]:


"""This generate the posting list from the given data"""
posting_list = {}
for i in range(len(G.nodes)):
    doc = G.nodes[i]['page_content']
    words = get_words(doc)
    for word in words:
        s = posting_list.get(word, None)
        if s is None:
            s = set()
            posting_list[word] = s
        s.add(i)


# In[12]:


def hits_algo(query):
    """ Words from the given query are lemmalizedand stored in qwords\n
    Base Set and Root set for the given query are generated\n
    Sub graph of the given graph is taken with nodes present in the BaseSet\n
    Adjacency matrix A is generated for the obtained subgraph\n
    Eigen vectors and Eigen values are generated for A.A^T and A^T.A \n
    The principal eigen vector of AA^T gives the hub scores and the principal eigen vector of A^T.A gives us the Authority scores of the nodes\n
    The nodes are then ordered according to their corresponding velues in the eigen vector\n
    This print the hubs and authorities in decreasing Scores\n


    :param
        This take given query as parameter



    :returns
        list of processing time and the number of edges
        
    """
    

    start = time.time()
    
    qwords = get_words(query)
    if(len(qwords)!=0): 
        posting_list.get(query,None)
        rootSet = posting_list.get(qwords[0],set()).copy()
        for word in qwords:
            rootSet = rootSet.intersection(posting_list.get(word,set()))
        baseSet = rootSet.copy()
        for node in rootSet:
            for i in range(len(Adj[0])):
                if Adj[node][i] == 1:
                    baseSet.add(i)
        for i in range(len(Adj)):
            for node in rootSet:
                if Adj[i][node] == 1:
                    baseSet.add(i)
        baseList = list(baseSet)
        G1 = G.subgraph(baseList)
        nodesList = list(G1.nodes)
        A  = nx.adjacency_matrix(G1)
        A = A.todense()
        A = np.array(A)
        AAT = A.dot(A.transpose())
        ATA = A.transpose().dot(A)
        hub,Hubscores = np.linalg.eig(AAT)
        index = np.argmax(hub)
        Hubscores[:,index]  = Hubscores[:,index]/((sum(Hubscores[:,index])))
        arr = np.flip((np.argsort(Hubscores[:,index])))
        auth,AuthorityScores = np.linalg.eig(ATA)
        index = np.argmax(auth)
        AuthorityScores[:,index] = AuthorityScores[:,index]/((sum(AuthorityScores[:,index])))
        arr1 = np.flip((np.argsort(AuthorityScores[:,index])))
        list_h = arr.tolist()
        orderOfHub = np.array(nodesList)
        orderOfHub[list_h]
        print("List of nodes with increasing HubScore: " )
        print(orderOfHub[list_h])
        list1 = arr1.tolist()
        orderOfAuthority = np.array(nodesList)
        orderOfAuthority[list1]
        print("List of nodes with increasing AuthorityScore: ")
        print (orderOfAuthority[list1])
        end = time.time()
        time_diff = end - start
        print(time_diff)
        #print("No of edges/links in the subgraph: ")
        print(len(G1.edges))
        return list([time_diff,len(G1.edges)])
    
    


# In[29]:


query = input("Enter the query : ")
hits_algo(query)


# In[ ]:





# In[39]:


i=1
dict1 ={}
for key in posting_list.keys():
    if(i>1400):
        break
    v = hits_algo(key)
    dict1[v[1]]=(v[0])
    i=i+1
    


# In[40]:


dict1 = collections.OrderedDict(sorted(dict1.items()))
dict1


# In[42]:




x = dict1.keys()
y = dict1.values()
 
# Plotting the Graph
plt.plot(x, y)
plt.title("Plot of Runtime vs Number of edges")
plt.xlabel("Number Of Edges")
plt.ylabel("Runtime")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




