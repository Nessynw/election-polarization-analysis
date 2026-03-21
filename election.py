import numpy as np
#qst 1 : génération aléatoire d'un profil de votantes
'''
n : nombre de votantes
m : nombre de candidats
polarization : degré de polarisation (0.0 à 1.0)
r : générateur aléatoire numpy
'''
def generate_profile(n: int, m: int, polarization: float = 0.0, r=None):
    if r is None:
        r = np.random.default_rng()
    a = r.integers(0, 2, size=m)
    a_op = 1 - a
    bruit = 0.5 * (1 - polarization)
    profile = np.empty((n, m), dtype=int)
    for i in range(n):
        ref = a if i < n // 2 else a_op
        noise = r.random(m) < bruit        # n → noise
        profile[i] = np.where(noise, 1 - ref, ref)
    return profile # profil de votantes (n, m) avec des votes binaires (0 ou 1)

# Example usage:
if __name__ == "__main__":
    r = np.random.default_rng(42)
    n, m = 8, 6

    for polarization in [0.0, 0.5, 1.0]:
        print(f"polarization: {polarization}")
        print(generale_profile(n, m, polarization, r))
        print()       
        