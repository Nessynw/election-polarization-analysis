import numpy as np
import itertools
import math
import random
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Question 1 : approbations
def generate_approval_profile(n, m, polarization=0.0, noise=0.0, r=None):
    """
    polarization=0 -> profil p_a (tous mêmes bulletins)
    polarization=1 -> profil p_{a,ā} (moitié a, moitié a_op)
    0<p<1        -> chaque votante choisit a_op avec probabilité p
    noise        -> probabilité de flip par bit (optionnel)
    """
    if n % 2 != 0 or m % 2 != 0:
        raise ValueError("n et m doivent être pairs")
    if r is None:
        r = np.random.default_rng()
 
    a = r.integers(0, 2, size=m)        # bulletin de base aléatoire
    a_op = 1 - a                         # bulletin opposé
 
    profile = np.empty((n, m), dtype=int)
    for i in range(n):
        # Choisir le bulletin de base selon le niveau de polarisation
        if polarization == 1.0 and i >= n // 2:
            base = a_op
        elif 0 < polarization < 1 and r.random() < polarization:
            base = a_op
        else:
            base = a
 
        # Appliquer le bruit bit à bit
        if noise > 0:
            flips = r.random(m) < noise
            profile[i] = np.where(flips, 1 - base, base)
        else:
            profile[i] = base.copy()
 
    return profile

# Question 2 : ordres totaux
def generate_rank_profile(n, m, polarization=0.0, noise=0.0, r=None):
    if n % 2 != 0 or m % 2 != 0:
        raise ValueError("n et m doivent être pairs")
    if r is None:
        r = np.random.default_rng()
 
    base = r.permutation(np.arange(1, m + 1))   # ordre de référence
    opposite = m - base + 1                       # ordre exactement opposé
 
    profile = np.empty((n, m), dtype=int)
    for i in range(n):
        if polarization == 1.0 and i >= n // 2:
            order = opposite
        elif 0 < polarization < 1 and r.random() < polarization:
            order = opposite
        else:
            order = base
 
        if noise > 0:
            k = max(1, int(round(noise * m)))
            tmp = order.copy()
            for _ in range(k):
                j = r.integers(0, m - 1)
                tmp[j], tmp[j + 1] = tmp[j + 1], tmp[j]
            profile[i] = tmp
        else:
            profile[i] = order.copy()
 
    return profile


# Question 3 : d_ck,cl
def pairwise_diffs_approval(profile: np.ndarray):
    n, m = profile.shape
    diffs = {}
    for i, j in itertools.combinations(range(m), 2):
        # Nombre de votantes préférant c_i à c_j (approuve i mais pas j)
        pref_i = np.sum((profile[:, i] == 1) & (profile[:, j] == 0))
        # Nombre de votantes préférant c_j à c_i
        pref_j = np.sum((profile[:, j] == 1) & (profile[:, i] == 0))
        diffs[(i, j)] = abs(int(pref_i - pref_j))
    return diffs
 

def pairwise_diffs_ranking(profile: np.ndarray):
    n, m = profile.shape
    diffs = {}
    for i, j in itertools.combinations(range(m), 2):
        # Nombre de votantes où c_i est mieux classée que c_j
        pref_i = np.sum(profile[:, i] < profile[:, j])
        pref_j = n - pref_i
        diffs[(i, j)] = abs(int(pref_i - pref_j))
    return diffs

# Question 5 : mesure de polarisation φ2
def _phi2_from_diffs(diffs: dict, n: int, m: int) -> float:
    """Calcule φ²(p) à partir des différences pairwise."""
    if m < 2 or n == 0:
        return 0.0
    total = sum(n - d for d in diffs.values())
    return total / (n * math.comb(m, 2))

def phi2_approval(profile: np.ndarray) -> float:
    """Calcule φ²(p) pour un profil de votes par approbation."""
    n, m = profile.shape
    return _phi2_from_diffs(pairwise_diffs_approval(profile), n, m)

def phi2_ranking(profile: np.ndarray) -> float:
    """Calcule φ²(p) pour un profil de votes par ordres totaux."""
    n, m = profile.shape
    return _phi2_from_diffs(pairwise_diffs_ranking(profile), n, m)

#question 6 : Evolution de φ2 en fonction de la polarisation
def plot_phi2_evolution(n: int, m: int):
    """
    Trace l'évolution de φ²(p) pour des votes par approbation et par ordres
    totaux en faisant varier le niveau de polarisation de 0 à 1.
    """
    polar_values = [round(p * 0.1, 1) for p in range(11)]
    phi2_app = []
    phi2_rank = []
    rng = np.random.default_rng()
 
    for p in polar_values:
        prof_a = generate_approval_profile(n, m, polarization=p, r=rng)
        prof_l = generate_rank_profile(n, m, polarization=p, r=rng)
        phi2_app.append(phi2_approval(prof_a))
        phi2_rank.append(phi2_ranking(prof_l))
 
    plt.figure(figsize=(8, 5))
    plt.plot(polar_values, phi2_app, marker='o', label="Approbation (φ²)")
    plt.plot(polar_values, phi2_rank, marker='s', label="Ordres totaux (φ²)")
    plt.xlabel("Niveau de polarisation")
    plt.ylabel("φ²(p)")
    plt.title("Évolution de φ² en fonction de la polarisation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("phi2_evolution.png", dpi=150)
    plt.show()

# Question 8 : distances hamming et Spearman
def hamming(a, b):
    if len(a) != len(b):
        raise ValueError("Taille incohérente")
    return int(np.sum(a != b))

def spearman(a, b):
    if len(a) != len(b):
        raise ValueError("Taille incohérente")
    return int(np.sum(np.abs(np.array(a) - np.array(b))))

# Question 12 : u1*(p)
def u1_approval(profile : np.ndarray) -> np.ndarray:
    n, m = profile.shape
    consensus = np.zeros(m, dtype=int)
    nb_ones = np.sum(profile, axis=0)
    consensus[nb_ones > n / 2] = 1
    return consensus

def cost_u1_approval(profile: np.ndarray) -> int:
    """Retourne la valeur optimale u*₁(p) pour des votes par approbation."""
    consensus = u1_approval(profile)
    return int(sum(hamming(profile[i], consensus) for i in range(len(profile))))

def u1_ranking(profile : np.ndarray) -> np.ndarray:
    n, m = profile.shape
    # Matrice de coût : costs[j, pos] = Σ_i |pos+1 - profile[i, j]|
    costs = np.zeros((m, m))
    for j in range(m):
        for pos in range(m):
            costs[j, pos] = np.sum(np.abs((pos + 1) - profile[:, j]))
 
    row_ind, col_ind = linear_sum_assignment(costs)
    consensus = np.zeros(m, dtype=int)
    consensus[row_ind] = col_ind + 1
    return consensus

def cost_u1_ranking(profile: np.ndarray) -> int:
    """Retourne la valeur optimale u*₁(p) pour des votes par ordres totaux."""
    consensus = u1_ranking(profile)
    return int(sum(spearman(profile[i], consensus) for i in range(len(profile))))
# Question 13 : u*2 estimé par k-means (k=2)
def consensus_approval(cluster : list) -> np.ndarray:
    if  not cluster:
        raise ValueError("Le cluster est vide")
    arr = np.array(cluster)
    return u1_approval(arr)

def consensus_ranking(cluster : list) -> np.ndarray:
    if not cluster:
        raise ValueError("Le cluster est vide")
    arr = np.array(cluster)
    return u1_ranking(arr)


def kmeans_approval(profile: np.ndarray, n_runs: int = 10) -> int:
    n = len(profile)
    if n < 2:
        raise ValueError("Le profil doit contenir au moins 2 votes")
    best_cost = float('inf')
    for _ in range(n_runs):
        #initialisation aléatoire de 2 profils
        idx1, idx2 = random.sample(range(n), 2)
        c1 = profile[idx1].copy()
        c2 = profile[idx2].copy()
        prev_labels = None
        
        for _ in range(100):
            labels = []
            for v in profile:
                d1 = hamming(v, c1)
                d2 = hamming(v, c2)
                labels.append(0 if d1 <= d2 else 1)
            if prev_labels is not None and labels == prev_labels:
                break  # convergence
            prev_labels = labels[:]
            cluster1 = [profile[i] for i in range(n) if labels[i] == 0]
            cluster2 = [profile[i] for i in range(n) if labels[i] == 1]
            if cluster1:
                c1 = consensus_approval(cluster1)
            if cluster2:
                c2 = consensus_approval(cluster2)
        #coût final
        cost = sum(
            hamming(profile[i], c1 if labels[i] == 0 else c2)
            for i in range(n)
        )
 
        if cost < best_cost:
            best_cost = cost
 
    return best_cost

def kmeans_ranking(profile: np.ndarray, n_runs: int = 10) -> int:
    n = len(profile)
    if n < 2:
        raise ValueError("Le profil doit contenir au moins 2 votes")
    
    best_cost = float('inf')
    
    for _ in range(n_runs):
        idx1, idx2 = random.sample(range(n), 2)
        c1 = profile[idx1].copy()
        c2 = profile[idx2].copy()
        
        prev_labels = None
        
        for _ in range(100):
            labels = []
            for v in profile:
                d1 = spearman(v, c1)
                d2 = spearman(v, c2)
                labels.append(0 if d1 <= d2 else 1)
            if prev_labels is not None and labels == prev_labels:
                break
            prev_labels = labels[:]
            cluster1 = [profile[i] for i in range(n) if labels[i] == 0]
            cluster2 = [profile[i] for i in range(n) if labels[i] == 1]
            if cluster1:
                c1 = consensus_ranking(cluster1)
            if cluster2:
                c2 = consensus_ranking(cluster2)
        cost = sum(
            spearman(profile[i], c1 if labels[i] == 0 else c2)
            for i in range(n)
        )
 
        if cost < best_cost:
            best_cost = cost
 
    return best_cost
# Question 14 : φ basé sur φ2
def phi_dH(profile: np.ndarray, n_runs: int = 10) -> float:
    """
    Calcule φ_{d_H}(p) = (2 / (n × m)) × (u*₁(p) − ũ*₂(p))
    pour des votes par approbation.
    """
    n, m = profile.shape
    u1 = cost_u1_approval(profile)
    u2 = kmeans_approval(profile, n_runs=n_runs)
    return (2 / (n * m)) * (u1 - u2)

def phi_dS(profile: np.ndarray, n_runs: int = 10) -> float:
    """
    Calcule φ_{d_S}(p) = (4 / (n × m²)) × (u*₁(p) − ũ*₂(p))
    pour des votes par ordres totaux.
    """
    n, m = profile.shape
    u1 = cost_u1_ranking(profile)
    u2 = kmeans_ranking(profile, n_runs=n_runs)
    return (4 / (n * m ** 2)) * (u1 - u2)

# Question 15 : évolution de φ_{d_H} et φ_{d_S} en fonction de la polarisation
def plot_phi_distance_evolution(n: int, m: int, n_runs: int = 10):
    """
    Trace l'évolution de φ_{d_H} et φ_{d_S} en faisant varier
    le niveau de polarisation de 0 à 1.
    """
    polar_values = [round(p * 0.1, 1) for p in range(11)]
    phi_dH_vals = []
    phi_dS_vals = []
    rng = np.random.default_rng()
 
    for p in polar_values:
        prof_a = generate_approval_profile(n, m, polarization=p, r=rng)
        prof_l = generate_rank_profile(n, m, polarization=p, r=rng)
        phi_dH_vals.append(phi_dH(prof_a, n_runs=n_runs))
        phi_dS_vals.append(phi_dS(prof_l, n_runs=n_runs))
        print(f"  polarisation={p:.1f}  φ_dH={phi_dH_vals[-1]:.4f}  "
              f"φ_dS={phi_dS_vals[-1]:.4f}")
 
    plt.figure(figsize=(8, 5))
    plt.plot(polar_values, phi_dH_vals, marker='o',
             label="Approbation (φ_{d_H})")
    plt.plot(polar_values, phi_dS_vals, marker='s',
             label="Ordres totaux (φ_{d_S})")
    plt.xlabel("Niveau de polarisation")
    plt.ylabel("Valeur de la mesure")
    plt.title("Évolution de φ_{d_H} et φ_{d_S} en fonction de la polarisation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("phi_distance_evolution.png", dpi=150)
    plt.show()
#main pour tester les différentes functions 
def main():
    rng = np.random.default_rng(42)

    # Test numérique simple
    profA = generate_approval_profile(n=10, m=4, polarization=0.7, noise=0.05, r=rng)
    profL = generate_rank_profile(n=10, m=4, polarization=0.7, noise=0.05, r=rng)
    print("phi2 approvals:", phi2_approval(profA))
    print("phi2 rankings :", phi2_ranking(profL))

    # Graphiques de validation 
    plot_phi2_evolution(n=40, m=4)            # produit phi2_evolution.png
    plot_phi_distance_evolution(n=40, m=4)    # produit phi_distance_evolution.png

if __name__ == "__main__":
    main()
