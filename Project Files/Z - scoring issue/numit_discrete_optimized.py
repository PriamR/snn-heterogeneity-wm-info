232.#!/usr/bin/env python3
"""
Numba-optimized version of numit_discrete.py

This version replaces the computational bottlenecks with JIT-compiled functions
for significant performance improvements in null model generation.
"""

import numpy as np
import numba as nb
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Boolean function lookup table (converted to array for numba)
BOOL_FUNC_NAMES = [
    'CONST_0', 'CONST_1', 'COPY_x', 'NOT_x', 'COPY_y', 'NOT_y',
    'AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR',
    'IMP_xy', 'IMP_yx', 'X_AND_NOTY', 'NOTX_AND_Y'
]

BOOL_FUNC_ARRAY = np.array([
    [0,0,0,0],  # CONST_0
    [1,1,1,1],  # CONST_1
    [0,0,1,1],  # COPY_x
    [1,1,0,0],  # NOT_x
    [0,1,0,1],  # COPY_y
    [1,0,1,0],  # NOT_y
    [0,0,0,1],  # AND
    [1,1,1,0],  # NAND
    [0,1,1,1],  # OR
    [1,0,0,0],  # NOR
    [0,1,1,0],  # XOR
    [1,0,0,1],  # XNOR
    [1,1,0,1],  # IMP_xy
    [1,0,1,1],  # IMP_yx
    [0,0,1,0],  # X_AND_NOTY
    [0,1,0,0],  # NOTX_AND_Y
], dtype=np.int8)

# Keep original dictionary for compatibility
BOOL_FUNCS = {
    'CONST_0': [0,0,0,0],
    'CONST_1': [1,1,1,1],
    'COPY_x': [0,0,1,1],
    'NOT_x':  [1,1,0,0],
    'COPY_y': [0,1,0,1],
    'NOT_y':  [1,0,1,0],
    'AND':    [0,0,0,1],
    'NAND':   [1,1,1,0],
    'OR':     [0,1,1,1],
    'NOR':    [1,0,0,0],
    'XOR':    [0,1,1,0],
    'XNOR':   [1,0,0,1],
    'IMP_xy': [1,1,0,1],  # x → y
    'IMP_yx': [1,0,1,1],  # y → x
    'X_AND_NOTY': [0,0,1,0],
    'NOTX_AND_Y': [0,1,0,0],
}

# Helper function to convert function name to index
def func_name_to_idx(name):
    return BOOL_FUNC_NAMES.index(name)

def apply_func(name, x, y):
    """Original function for compatibility"""
    table = BOOL_FUNCS[name]
    return table[2*x + y]

@nb.njit
def apply_func_numba(func_idx, x, y):
    """Numba-optimized boolean function application"""
    return BOOL_FUNC_ARRAY[func_idx, 2*x + y]

@nb.njit
def build_pi_numba(pX, pY):
    """Numba-optimized pi distribution builder"""
    pi = np.zeros(4)
    pi[0] = (1-pX)*(1-pY)  # (0,0)
    pi[1] = (1-pX)*pY      # (0,1)
    pi[2] = pX*(1-pY)      # (1,0)
    pi[3] = pX*pY          # (1,1)
    return pi

def build_pi(pX, pY):
    """Original function - now calls numba version"""
    return build_pi_numba(pX, pY)

@nb.njit
def mutual_info_numba(P):
    """Numba-optimized mutual information calculation"""
    # Compute marginals
    Pi = np.zeros(4)
    Pj = np.zeros(4)
    
    for i in range(4):
        for j in range(4):
            Pi[i] += P[i,j]
            Pj[j] += P[i,j]
    
    # Compute MI
    I = 0.0
    for i in range(4):
        for j in range(4):
            pij = P[i,j]
            if pij > 0:
                I += pij * np.log2(pij / (Pi[i] * Pj[j]))
    return I

def mutual_info(P):
    """Original function - now calls numba version"""
    return mutual_info_numba(P)

@nb.njit
def joint_matrix_numba(pX, pY, f1_idx, f2_idx, eta):
    """Numba-optimized joint matrix construction"""
    pi = build_pi_numba(pX, pY)
    P = np.zeros((4, 4))
    noise_dist = np.ones(4) / 4.0  # uniform noise distribution
    
    for x in range(2):
        for y in range(2):
            i = 2*x + y
            u = apply_func_numba(f1_idx, x, y)
            v = apply_func_numba(f2_idx, x, y)
            j_det = 2*u + v
            
            for j in range(4):
                if j == j_det:
                    prob = (1-eta) + eta*noise_dist[j]
                else:
                    prob = eta*noise_dist[j]
                P[i,j] += pi[i]*prob
    
    return P

def joint_matrix_old(pX, pY, f1, f2, eta):
    """Original function - kept for compatibility"""
    pi = build_pi(pX, pY)
    P = np.zeros((4,4))
    for x in (0,1):
        for y in (0,1):
            i = 2*x + y
            u = apply_func(f1, x, y)
            v = apply_func(f2, x, y)
            for dx in (0,1):
                for dy in (0,1):
                    prob_flip = (eta**(dx+dy)) * ((1-eta)**(2-(dx+dy)))
                    j = 2*(u ^ dx) + (v ^ dy)
                    P[i,j] += pi[i]*prob_flip
    return P

def joint_matrix(pX, pY, f1, f2, eta, noise_dist=None):
    """
    Optimized joint matrix construction.
    Falls back to numba version when possible, original when using custom noise.
    """
    if noise_dist is None:
        # Use numba-optimized version
        f1_idx = func_name_to_idx(f1)
        f2_idx = func_name_to_idx(f2)
        return joint_matrix_numba(pX, pY, f1_idx, f2_idx, eta)
    else:
        # Use original version for custom noise distributions
        if noise_dist is None:
            noise_dist = np.ones(4) / 4.0
        else:
            noise_dist = np.array(noise_dist, dtype=float)
            noise_dist = noise_dist / noise_dist.sum()

        pi = build_pi(pX, pY)
        P = np.zeros((4,4))

        for x in (0,1):
            for y in (0,1):
                i = 2*x + y
                u = apply_func(f1, x, y)
                v = apply_func(f2, x, y)
                j_det = 2*u + v
                for j in range(4):
                    prob = (1-eta if j == j_det else 0.0) + eta*noise_dist[j]
                    P[i,j] += pi[i]*prob
        return P

@nb.njit
def solve_eta_numba(I_target, pX, pY, f1_idx, f2_idx, tol=1e-6):
    """Numba-optimized eta solving with bisection"""
    lo, hi = 0.0, 0.5
    
    # Compute boundary MIs
    P_lo = joint_matrix_numba(pX, pY, f1_idx, f2_idx, lo)
    P_hi = joint_matrix_numba(pX, pY, f1_idx, f2_idx, hi)
    Ilo = mutual_info_numba(P_lo)
    Ihi = mutual_info_numba(P_hi)
    
    # Check feasibility
    if not (min(Ilo, Ihi) <= I_target <= max(Ilo, Ihi)):
        return -1.0, np.zeros((4, 4))  # Return invalid eta to signal failure
    
    # Bisection
    for _ in range(60):
        mid = 0.5*(lo + hi)
        P_mid = joint_matrix_numba(pX, pY, f1_idx, f2_idx, mid)
        Imid = mutual_info_numba(P_mid)
        
        if abs(Imid - I_target) < tol:
            return mid, P_mid
        
        if Imid > I_target:
            lo = mid
        else:
            hi = mid
    
    # Return final estimate
    eta = 0.5*(lo + hi)
    P_final = joint_matrix_numba(pX, pY, f1_idx, f2_idx, eta)
    return eta, P_final

def solve_eta(I_target, pX, pY, f1, f2, tol=1e-6):
    """
    Optimized solve_eta - uses numba version when possible
    """
    try:
        f1_idx = func_name_to_idx(f1)
        f2_idx = func_name_to_idx(f2)
        eta, P = solve_eta_numba(I_target, pX, pY, f1_idx, f2_idx, tol)
        if eta >= 0:  # Valid solution found
            return eta, P
    except (ValueError, IndexError):
        pass  # Fall back to original method
    
    # Original implementation as fallback
    lo, hi = 0.0, 0.5
    Ilo = mutual_info(joint_matrix(pX,pY,f1,f2,lo))
    Ihi = mutual_info(joint_matrix(pX,pY,f1,f2,hi))
    if not (min(Ilo,Ihi) <= I_target <= max(Ilo,Ihi)):
        return None, None
    for _ in range(60):
        mid = 0.5*(lo+hi)
        Imid = mutual_info(joint_matrix(pX,pY,f1,f2,mid))
        if abs(Imid - I_target) < tol:
            return mid, joint_matrix(pX,pY,f1,f2,mid)
        if Imid > I_target:
            lo = mid
        else:
            hi = mid
    eta = 0.5*(lo+hi)
    return eta, joint_matrix(pX,pY,f1,f2,eta)

def find_transition_for_target(I_target, max_tries=500):
    """
    Optimized find_transition_for_target using numba functions when possible
    """
    funcs = list(BOOL_FUNCS.keys())
    for _ in range(max_tries):
        f1 = random.choice(funcs)
        f2 = random.choice(funcs)
        pX, pY = np.random.rand(), np.random.rand()
        
        # Try numba-optimized version first
        try:
            f1_idx = func_name_to_idx(f1)
            f2_idx = func_name_to_idx(f2)
            P_det = joint_matrix_numba(pX, pY, f1_idx, f2_idx, 0.0)
            I_det = mutual_info_numba(P_det)
            
            if I_det < I_target:
                continue
                
            eta, P = solve_eta_numba(I_target, pX, pY, f1_idx, f2_idx)
            if eta >= 0:  # Valid solution
                return f1, f2, eta, P
        except (ValueError, IndexError):
            # Fall back to original method
            I_det = mutual_info(joint_matrix(pX,pY,f1,f2,0.0))
            if I_det < I_target:
                continue
            eta, P = solve_eta(I_target, pX, pY, f1, f2)
            if eta is not None:
                return f1, f2, eta, P
    
    raise RuntimeError("No suitable (f1,f2) combination found in max_tries")


@nb.njit
def build_pi_multivariate_numba(p_vars):
    """Numba-optimized multivariate pi distribution builder"""
    n_vars = len(p_vars)
    K = 1 << n_vars
    pi = np.zeros(K)
    for s in range(K):
        prob = 1.0
        for i in range(n_vars):
            bit = (s >> (n_vars - 1 - i)) & 1
            if bit == 1:
                prob *= p_vars[i]
            else:
                prob *= (1.0 - p_vars[i])
        pi[s] = prob
    return pi

@nb.njit
def joint_matrix_multivariate_numba(p_vars, funcs, eta, noise_dist):
    """Numba-optimized multivariate joint matrix construction"""
    n_vars = len(p_vars)
    K = 1 << n_vars
    pi = build_pi_multivariate_numba(p_vars)
    P = np.zeros((K, K))
    
    for i in range(K):
        if pi[i] == 0:
            continue
        j_det = 0
        for v in range(n_vars):
            bit = funcs[v, i]
            j_det = (j_det << 1) | bit
            
        for j in range(K):
            if j == j_det:
                prob = (1.0 - eta) + eta * noise_dist[j]
            else:
                prob = eta * noise_dist[j]
            P[i, j] += pi[i] * prob
            
    return P

@nb.njit
def mutual_info_multivariate_numba(P):
    """Numba-optimized multivariate mutual information calculation"""
    K = P.shape[0]
    Pi = np.zeros(K)
    Pj = np.zeros(K)
    
    for i in range(K):
        for j in range(K):
            Pi[i] += P[i, j]
            Pj[j] += P[i, j]
            
    I = 0.0
    for i in range(K):
        for j in range(K):
            pij = P[i, j]
            if pij > 0:
                I += pij * np.log2(pij / (Pi[i] * Pj[j]))
    return I

@nb.njit
def solve_eta_multivariate_numba(I_target, p_vars, funcs, noise_dist, tol=1e-6):
    """Numba-optimized multivariate eta solving with bisection"""
    lo, hi = 0.0, 1.0  # Use 1.0 to allow completely uniform noise
    
    P_lo = joint_matrix_multivariate_numba(p_vars, funcs, lo, noise_dist)
    P_hi = joint_matrix_multivariate_numba(p_vars, funcs, hi, noise_dist)
    Ilo = mutual_info_multivariate_numba(P_lo)
    Ihi = mutual_info_multivariate_numba(P_hi)
    
    if not (min(Ilo, Ihi) <= I_target <= max(Ilo, Ihi)):
        K = 1 << len(p_vars)
        return -1.0, np.zeros((K, K))
        
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        P_mid = joint_matrix_multivariate_numba(p_vars, funcs, mid, noise_dist)
        Imid = mutual_info_multivariate_numba(P_mid)
        
        if abs(Imid - I_target) < tol:
            return mid, P_mid
            
        if Imid > I_target:
            lo = mid
        else:
            hi = mid
            
    eta = 0.5 * (lo + hi)
    P_final = joint_matrix_multivariate_numba(p_vars, funcs, eta, noise_dist)
    return eta, P_final

def find_transition_for_target_multivariate(I_target, n_vars, max_tries=500, tol=1e-6):
    """
    Finds a multivariate transition matrix that matches a target mutual information.
    
    Args:
        I_target (float): Target mutual information.
        n_vars (int): Number of binary variables (dimension).
        max_tries (int): Maximum number of random function combinations to try.
        tol (float): Tolerance for the mutual information matching.
        
    Returns:
        tuple: (funcs, p_vars, eta, P) where:
            - funcs: (n_vars, 2**n_vars) array of boolean functions.
            - p_vars: array of n_vars probabilities for the independent source variables.
            - eta: mixing parameter.
            - P: (2**n_vars, 2**n_vars) joint probability matrix.
    """
    K = 1 << n_vars
    noise_dist = np.ones(K) / K
    
    for _ in range(max_tries):
        p_vars = np.random.rand(n_vars)
        # Randomly draw boolean functions for each target variable
        funcs = np.random.randint(0, 2, size=(n_vars, K)).astype(np.int8)
        
        P_det = joint_matrix_multivariate_numba(p_vars, funcs, 0.0, noise_dist)
        I_det = mutual_info_multivariate_numba(P_det)
        
        if I_det < I_target:
            continue
            
        eta, P = solve_eta_multivariate_numba(I_target, p_vars, funcs, noise_dist, tol=tol)
        if eta >= 0:
            return funcs, p_vars, eta, P
            
    raise RuntimeError("No suitable multivariate functions found in max_tries")
