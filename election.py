import numpy as np
from scipy.optimize import linear_sum_assignment
#qst 1 : génération aléatoire d'un profil de votantes par approbation
'''
chaque votante exprime ses préférences en approuvant (1) ou non (0)
n : nombre de votantes
m : nombre de candidats
polarization : degré de polarisation (0.0: p^a à 1.0 : p^{a,ā})
r : générateur aléatoire numpy
'''
def generate_profile(n: int, m: int, polarization: float = 0.0, r=None):
    if r is None:
        r = np.random.default_rng()
    
    a = r.integers(0, 2, size=m)
    a_op = 1 - a
    
    profile = np.empty((n, m), dtype=int)
    for i in range(n):
        # faible polarisation --> tous votent a
        if polarization == 0.0:
            profile[i] = a
         # forte polarisation --> moitié a, moitié opposé
        elif polarization == 1.0:
            profile[i] = a if i < n // 2 else a_op
        else:
            # Polarisation intermédiaire 
            profile[i] = a_op if r.random() < polarization else a

    return profile
#qst 2 : génération aléatoire d'un profil de votantes par ordre de préférence
'''
Chaque votante exprime ses préférences en classant les candidats de la plus préférée (1) à la moins préféréem(m)
n : nombre de votantes
m : nombre de candidats
polarization : degré de polarisation (0.0: p^a à 1.0 : p^{a,ā})
r : générateur aléatoire numpy
'''
def generate_rank_profile(n:int, m:int, polarization:float=0.0, r=None):
    if r is None:
        r = np.random.default_rng()
    
    order = r.permutation(np.arange(1,m+1))
    order_op = m - order + 1
    
    profile = np.empty((n, m), dtype=int)
    for i in range(n):
        if polarization == 0.0:
            profile[i] = order
        elif polarization == 1.0:
            profile[i] = order if i < n // 2 else order_op
        else:
            profile[i] = order_op if r.random() < polarization else order

    return profile
# Example usage:
if __name__ == "__main__":
    r = np.random.default_rng(42)
    n, m = 8, 6

    for polarization in [0.0, 0.5, 1.0]:
        print(f"polarization: {polarization}")
        print(generate_profile(n, m, polarization, r))
        print()      
        print(f"polarization: {polarization}")
        print(generate_rank_profile(n, m, polarization, r))
        print()

#question8
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


#question12
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
