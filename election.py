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

# Example usage:
if __name__ == "__main__":
    r = np.random.default_rng(42)
    n, m = 8, 6

    for polarization in [0.0, 0.5, 1.0]:
        print(f"polarization: {polarization}")
        print(generate_profile(n, m, polarization, r))
        print()      
        