import numpy as np
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