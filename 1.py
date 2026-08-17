import numpy as np
import scipy as sc

corpus = {
    'the cats like the eggs',
     'I hate cats',
     'I like eggs'
     }


splited_list = []
for document in corpus:
    lowered = document.lower()
    n = lowered.split()
    splited_list.append(n)
    

vec_num = {
    "i": 0,
    "like": 1,
    "eggs": 2,
    "hate": 3,
    "cats": 4,
    "the": 5
}



# ----
#      i,like,eggs,hate,cats,the
# doc1,0,1,1,0,1,2
# doc2,1,0,0,1,1,0
# doc3,1,1,1,0,0,0

# ----

m = np.zeros([3,6])
for i in range(0,3):
    for docs in splited_list[i]:
        m[i,vec_num[docs]] += 1
print(m)
        
    

