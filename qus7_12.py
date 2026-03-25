import numpy as np
from scipy.optimize import linear_sum_assignment

def hamming(a,b): #a et b sont des bulletins de taille m (vote par approbations)
    if (len(a)!=len(b)):
        print("Un ou plusieurs bulletins n'ont pas la bonne taille")
        return
    d=0
    for i in range(len(a)):
        if (a[i]!=b[i]):
            d+=1
    return d

def spearman(a,b): # (vote par ordres totaux)
    if (len(a)!=len(b)):
        print("Un ou plusieurs bulletins n'ont pas la bonne taille")
        return
    d=0
    for i in range(len(a)):
        d+=abs(r(a)[i]-r(b)[i]) #r(a) est le vecteur de taille m qui pour chaque candidate i associe son rang
    return d

def u1ParApprobation(profile):
    n=len(profile)
    m=len(profile[0])
    sol=np.zeros(m) #bulletin solution de u1* de taille m
    nb1=np.zeros(m) #somme de chaque bulletin
    for j in range(m):
        for i in range(n):
            nb1[j]+=profile[i,j]
        if (nb1[j]<=n/2): #si pour le candidat j il y a plus de votes 0 que de votes 1
            sol[j]=0
        else:
            sol[j]=1
    return sol

def u1ParOrdresTotaux(profile):
    n=len(profile)
    m=len(profile[0])
    sol=np.zeros(m) #bulletin solution de u1* de taille m
    couts=np.zeros((m,m)) #matrice des poids
    for j in range(m):
        for pos in range(m):
            couts[j,pos]=np.sum(np.abs(pos+1-profile[:,j]))
    row_ind, col_ind = linear_sum_assignment(couts) #pour chaque indice de ligne rowind [0,1,2,..m] on associe l'indice de colonne optimal colind
    sol[row_ind]=col_ind+1
    return sol

profilApprobations=np.array([[1,0,0,1],[0,1,0,1],[1,0,0,1],[0,1,1,1]])
print(u1ParApprobation(profilApprobations))

profilOrdresTotaux=np.array([[1,3,2,4],[1,2,3,4],[4,1,2,3],[1,2,3,4]])
print(u1ParOrdresTotaux(profilOrdresTotaux))
