# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:39:20 2019

@author: yvonn
"""

import pandas as pd
import math
#import numpy as np
#from operator import itemgetter

data = pd.read_excel(r'D:\Dataset_number_4.xlsx')
s = [[0]*12 for i in range(100)]        # Create a list
sm1 = [0]*10
#s=np.arange(1200.0).reshape(100,12)      # Create an array with float numbers
m = 100

# =============================================================================
# s.append(data['Feature0'])
# s.append(data['Feature1'])
# s.append(data['Feature2'])
# s.append(data['Feature3'])
# s.append(data['Feature4'])
# s.append(data['Feature5'])
# s.append(data['Feature6'])
# s.append(data['Feature7'])
# s.append(data['Feature8'])
# s.append(data['Feature9'])
# s.append(data['Label'])
# s.append(data['Distribution'])
# =============================================================================
#print(data['Feature0'][10])  #先列后行

#x= [[8, 9, 7],
#       [1, 2, 3],
#       [5, 4, 3],
#       [4, 5, 6]]
#x.sort(key=lambda x:x[0])
#print(x[:,0])


for i in range(100):           # Give the value of data (read from excel) to list s
    s[i] = data.iloc[i]

 
#s.sort(key=lambda x:x[0])               # Sort a list
#print(s[2])                             # Print third row of list
#print ([col[0] for col in s])           # Print first column of list


theta = float()  # Declare a float number
j = int()
j_1 = int()
d = 10
#F = float('inf')
F_1 = math.inf

for j in range (d):
    s.sort(key=lambda x:x[j])
    #s=sorted(s,key=itemgetter(0))
    #print (s)
    sm1[j] = s[m-1][j] + 1
    F = 0
    for i in range(m):
        if (s[i][10] == 1):
            F = F + s[i][11]
    if (F < F_1):
        F_1 = F
        theta = s[0][j] - 1
        j_1 = j
    for i in range(m):
        F = F - s[i][10] * s[i][11]
        if (i < m-1):
            if (F < F_1 and s[i][j] != s[i + 1][j]):
                F_1 = F
                theta = 1.0/2.0*(s[i][j] + s[i+1][j])
                j_1 = j
        elif (i == m-1):
            if (F < F_1 and s[i][j] != sm1[j]):
                F_1 = F
                theta = 1.0/2.0*(s[i][j] + sm1[j])
                j_1 = j

print('j* is',j_1,'\n','theta* is',theta)
