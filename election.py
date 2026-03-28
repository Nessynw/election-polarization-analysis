import numpy as np
import itertools
import math
import random
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Question 1 : approbations
def generate_approval_profile(n, m, polarization=0.0, noise=0.0, r=None):
    '''Génère un profil de votes par approbation avec n votantes et m candidates.'''
# n et m doivent être pairs pour garantir une polarisation symétrique.
    if n % 2 != 0 or m % 2 != 0:
        raise ValueError("n et m doivent être pairs")
    if r is None:
        r = np.random.default_rng() #géneration de nombres aléatoires

    a    = r.integers(0, 2, size=m) # profil de base aléatoire (0 ou 1 pour chaque candidate)
    a_op = 1 - a # profil opposé (inversion de 0 et 1)

    profile = np.empty((n, m), dtype=int) # matrice pour stocker les votes de chaque votante (n lignes, m colonnes)
    for i in range(n):
        if polarization == 1.0: 
            base = a_op if i >= n // 2 else a   
        elif polarization > 0:
            base = a_op if r.random() < polarization / 2 else a 
        else:
            base = a
        # Ajout de bruit : chaque vote a une probabilité "noise" d'être inversé
        if noise > 0:
            flips = r.random(m) < noise
            profile[i] = np.where(flips, 1 - base, base)
        else:
            profile[i] = base.copy()
    return profile # matrice finale des votes par approbation, avec n votantes et m candidates


def generate_rank_profile(n, m, polarization=0.0, noise=0.0, r=None):
    '''Génère un profil de votes par ordres totaux avec n votantes et m candidates.'''
    if n % 2 != 0 or m % 2 != 0:
        raise ValueError("n et m doivent être pairs")
    if r is None:
        r = np.random.default_rng()

    base     = r.permutation(np.arange(1, m + 1)) # ordre de base aléatoire 
    opposite = m - base + 1 # ordre opposé (inversion de la position des candidates)

    profile = np.empty((n, m), dtype=int) 
    for i in range(n):
        if polarization == 1.0:
            order = opposite if i >= n // 2 else base  
        elif polarization > 0:
            order = opposite if r.random() < polarization / 2 else base  
        else:
            order = base
# Ajout de bruit : chaque position a une probabilité "noise" d'être échangée avec la suivante
        if noise > 0:
            k = max(1, int(round(noise * m))) 
            tmp = order.copy() # copie de l'ordre pour pouvoir le modifier sans affecter les autres votantes
            for _ in range(k):
                j = r.integers(0, m - 1)
                tmp[j], tmp[j + 1] = tmp[j + 1], tmp[j]
            profile[i] = tmp
        else:
            profile[i] = order.copy()
    return profile

# Question 3 : d_ck,cl
def pairwise_diffs_approval(profile: np.ndarray):
    '''Calcule les différences pairwise pour un profil de votes par approbation.'''
    n, m = profile.shape # nombre de votantes (n) et de candidates (m)
    diffs = {} # dictionnaire pour stocker les différences entre chaque paire de candidates (i, j)
    for i, j in itertools.combinations(range(m), 2): 
        # Nombre de votantes préférant c_i à c_j (approuve i mais pas j)
        pref_i = np.sum((profile[:, i] == 1) & (profile[:, j] == 0))
        # Nombre de votantes préférant c_j à c_i
        pref_j = np.sum((profile[:, j] == 1) & (profile[:, i] == 0))
        diffs[(i, j)] = abs(int(pref_i - pref_j))
    return diffs
 

def pairwise_diffs_ranking(profile: np.ndarray):
    '''Calcule les différences pairwise pour un profil de votes par ordres totaux.'''
    n, m = profile.shape
    diffs = {}
    for i, j in itertools.combinations(range(m), 2):
        # Nombre de votantes où c_i est mieux classée que c_j
        pref_i = np.sum(profile[:, i] < profile[:, j])
        # Nombre de votantes où c_j est mieux classée que c_i
        pref_j = np.sum(profile[:, j] < profile[:, i])
        diffs[(i, j)] = abs(int(pref_i - pref_j))
    return diffs

# Question 5 : mesure de polarisation φ2
def _phi2_from_diffs(diffs: dict, n: int, m: int) -> float:
    """Calcule φ²(p) à partir des différences pairwise."""
    if m < 2 or n == 0: # pas de paires ou pas de votantes, la polarisation est nulle
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
def plot_phi2_evolution(n: int, m: int, nb_runs: int = 20, seed: int = 42):
    polar_values = [round(p * 0.1, 1) for p in range(11)]
    phi2_app  = []
    phi2_rank = []

    for p in polar_values:
        vals_a, vals_l = [], []
        for run in range(nb_runs):
            rng = np.random.default_rng(seed + run)
            vals_a.append(phi2_approval(generate_approval_profile(n, m, polarization=p, r=rng)))
            rng = np.random.default_rng(seed + run + 1000)
            vals_l.append(phi2_ranking(generate_rank_profile(n, m, polarization=p, r=rng)))
        phi2_app.append(np.mean(vals_a))
        phi2_rank.append(np.mean(vals_l))

    plt.figure(figsize=(9, 5))
    plt.plot(polar_values, phi2_app,  marker='o', label="φ² approbation",   color='steelblue')
    plt.plot(polar_values, phi2_rank, marker='s', label="φ² ordres totaux", color='orange')
    plt.xlabel("Niveau de polarisation (0 = peu polarisé, 1 = très polarisé)")
    plt.ylabel("φ²")
    plt.title("Evolution de φ² en fonction de la polarisation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("phi2_evolution.png", dpi=150)
    print("Figure sauvegardée : phi2_evolution.png")
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
    nb_ones = np.sum(profile, axis=0) #on somme chaque colonne
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
    n, m = profile.shape
    u1 = cost_u1_approval(profile)
    u2 = kmeans_approval(profile, n_runs=n_runs)
    return max(0.0, (2 / (n * m)) * (u1 - u2))

def phi_dS(profile: np.ndarray, n_runs: int = 10) -> float:
    n, m = profile.shape
    u1 = cost_u1_ranking(profile)
    u2 = kmeans_ranking(profile, n_runs=n_runs)
    return max(0.0, (4 / (n * m ** 2)) * (u1 - u2))

# Question 15 : évolution de φ_{d_H} et φ_{d_S} en fonction de la polarisation
"""
    Trace l'évolution de φ_{d_H} et φ_{d_S} en faisant varier
    le niveau de polarisation de 0 à 1.
"""
def plot_phi_distance_evolution(n: int, m: int, n_runs: int = 10, nb_runs: int = 8, seed: int = 42):
    polar_values = [round(p * 0.05, 2) for p in range(21)]
    phi_dH_means, phi_dH_stds = [], []
    phi_dS_means, phi_dS_stds = [], []

    for p in polar_values:
        vals_dH, vals_dS = [], []
        for run in range(nb_runs):
            rng = np.random.default_rng(seed + run)
            vals_dH.append(phi_dH(generate_approval_profile(n, m, polarization=p, r=rng), n_runs=n_runs))
            rng = np.random.default_rng(seed + run + 1000)
            vals_dS.append(phi_dS(generate_rank_profile(n, m, polarization=p, r=rng), n_runs=n_runs))
        phi_dH_means.append(np.mean(vals_dH))
        phi_dH_stds.append(np.std(vals_dH))
        phi_dS_means.append(np.mean(vals_dS))
        phi_dS_stds.append(np.std(vals_dS))

    phi_dH_means = np.array(phi_dH_means) #moyennes pour phi_dH
    phi_dH_stds  = np.array(phi_dH_stds) #ecarts-types
    phi_dS_means = np.array(phi_dS_means)
    phi_dS_stds  = np.array(phi_dS_stds)

    plt.figure(figsize=(9, 5))
    plt.plot(polar_values, phi_dH_means, marker='o', color='steelblue', label="φ_dH (approbation)") #courbe des moyennes de phi
    plt.fill_between(polar_values, phi_dH_means - phi_dH_stds, phi_dH_means + phi_dH_stds,
                     alpha=0.2, color='steelblue') #construction de l'intervalle d'ecart-type
    plt.plot(polar_values, phi_dS_means, marker='s', color='orange', label="φ_dS (ordres totaux)")
    plt.fill_between(polar_values, phi_dS_means - phi_dS_stds, phi_dS_means + phi_dS_stds,
                     alpha=0.2, color='orange')
    plt.xlabel("Niveau de polarisation (0 = peu polarisé, 1 = très polarisé)")
    plt.ylabel("Mesure de polarisation")
    plt.title(f"Comparaison φ_dH et φ_dS  (n={n}, m={m}, {nb_runs} répétitions)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("phi_dH_dS_comparison.png", dpi=150)
    print("Figure sauvegardée : phi_dH_dS_comparison.png")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Évolution de φ_dH et φ_dS en fonction du niveau de polarisation\n"
        f"(n={n} votantes, m={m} candidates, {nb_runs} répétitions par niveau)", fontsize=12)
    ax1.plot(polar_values, phi_dH_means, marker='o', color='steelblue', label="φ_dH moyen")
    ax1.fill_between(polar_values, phi_dH_means - phi_dH_stds, phi_dH_means + phi_dH_stds,
                     alpha=0.2, color='steelblue', label="±1 écart-type")
    ax1.set_title("φ_dH – Votes par approbation")
    ax1.set_xlabel("Niveau de polarisation (0 = peu polarisé, 1 = très polarisé)")
    ax1.set_ylabel("φ_dH(p)")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(polar_values, phi_dS_means, marker='s', color='orange', label="φ_dS moyen")
    ax2.fill_between(polar_values, phi_dS_means - phi_dS_stds, phi_dS_means + phi_dS_stds,
                     alpha=0.2, color='orange', label="±1 écart-type")
    ax2.set_title("φ_dS – Votes par ordres totaux")
    ax2.set_xlabel("Niveau de polarisation (0 = peu polarisé, 1 = très polarisé)")
    ax2.set_ylabel("φ_dS(p)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("phi_dH_dS_evolution.png", dpi=150)
    print("Figure sauvegardée : phi_dH_dS_evolution.png")
    plt.show()


#main pour tester les différentes functions 
def main():
    rng = np.random.default_rng(42)

    # Tests numériques
    profA = generate_approval_profile(n=10, m=4, polarization=0.7, noise=0.05, r=rng)
    profL = generate_rank_profile(n=10, m=4, polarization=0.7, noise=0.05, r=rng)
    print("phi2 approvals:", phi2_approval(profA))
    print("phi2 rankings :", phi2_ranking(profL))

   #graphiques 
    plot_phi2_evolution(n=40, m=4, nb_runs=20, seed=42)
    plot_phi_distance_evolution(n=40, m=4, n_runs=20, nb_runs=20, seed=42)

if __name__ == "__main__":
    main()