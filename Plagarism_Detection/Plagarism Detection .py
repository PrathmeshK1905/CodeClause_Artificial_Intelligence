#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


from sklearn.metrics.pairwise import cosine_similarity


# In[4]:


student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]


# In[5]:


student_files


# In[6]:


student_notes = [open(_file,encoding='utf-8').read() for _file in student_files]


# In[7]:


student_notes


# In[8]:


def vectorize(text):
    return TfidfVectorizer().fit_transform(text).toarray()


# In[9]:


def similarity(doc1,doc2):
    return cosine_similarity([doc1,doc2])


# In[10]:


vectors = vectorize(student_notes)


# In[11]:


vectors


# In[12]:


s_vectors = list(zip(student_files,vectors))


# In[13]:


s_vectors


# In[14]:


plagarism_results = set()


# In[15]:


def check_plagiarism():
    global s_vectors
    for student_a,text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a,text_vector_a))
        print(current_index)
        del new_vectors[current_index]
        for student_b,text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a,text_vector_b)[0][1]
            print(similarity(text_vector_a,text_vector_b)[0][1])
            student_pair = sorted((student_a,student_b))
            #print(similarity(text_vector_a,text_vector_b))
            #print(student_pair)
            score = (student_pair[0],student_pair[1],sim_score)
            plagarism_results.add(score)
    return plagarism_results


# In[16]:


for data in check_plagiarism():
    print(data)


# In[ ]:




