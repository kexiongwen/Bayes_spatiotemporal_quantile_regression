
import torch
from torch import einsum
from torch.distributions.chi2 import Chi2
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.linalg import cholesky, solve, solve_triangular

def inv_gauss(mu):
        
    ink = mu * torch.randn_like(mu).pow(2)
    a = 1 + 0.5 * (ink - ((ink + 2).square() - 4).sqrt())
    
    return torch.where((1 / (1 + a)) >= torch.rand_like(mu), mu * a, mu / a)

def generate_A(P: int, n: int, device = None, dtype = torch.float64):
    
    if n < P:
        raise ValueError(f"n={n} must be >= P={P}")
    
    A = torch.zeros(P, P, device = device, dtype = dtype)
    
    # Sub-diagonal entries: N(0,1)
    
    tril_indices = torch.tril_indices(P, P, offset = -1, device = device)
    A[tril_indices[0], tril_indices[1]] = torch.randn(tril_indices[0].shape[0], device = device, dtype = dtype)
    
    # Diagonal entries: sqrt(chi2) with df = n, n-1, ..., n-P+1
    
    df = torch.arange(n, n - P, -1, device = device, dtype = dtype)   # shape (P,)
    chi2 = Chi2(df).sample()             # shape (P,)
    A.diagonal().copy_(torch.sqrt(chi2))
    
    return A

def Sample_IW(V, n):
    """
    Sample inverse Wishart distribution with scatter matrix V and the degrees of freedom n,
    using the Bartlett decomposition.

    Args:
        V: scatter matrix V, shape (p, p)
        n: degrees of freedom (must be >= p)

    Returns:
        X_inv: symmetric positive definite matrix of shape (P, P)
    """
    
    L = cholesky(V)
    P = L.shape[0]
    device = L.device
    dtype = L.dtype

    # 1. Generate A
    A = generate_A(P, n, device = device, dtype = dtype)

    # 2. Compute L_inv and A_inv using triangular solves
    I = torch.eye(P, device = device, dtype = dtype)
    L_inv = solve_triangular(L, I, upper = False, unitriangular = False)
    
    # 3. R = A_inv @ L_inv (both lower triangular, product remains lower triangular)
    R = solve_triangular(A, L_inv, upper = False, unitriangular = False)

    return R.T @ R

def batch_mvn_sampler(Cov):
    """
    Args:
        covariance_matrices: Tensor of shape (T, P, P)
        num_samples: Number of samples to draw per batch
    
    Returns:
        samples: Tensor of shape (T, P)
    """
    
    T, P, _ = Cov.shape
    
    dtype = Cov.dtype
    
    # Compute Cholesky decomposition for each covariance matrix
    L = cholesky(Cov)
    
    # Generate standard normal samples
    z = torch.randn(T, P, 1, dtype = dtype, device = Cov.device)
   
    return torch.bmm(L, z).squeeze(-1)


def FFBS(Y, F, G, U, phi, W, a, b):
    """
    Args:        
        Y: Tensor of shape (T,)
        F: Tensor of shape (T, P)
        G: Tensor of shape (P, P)
        U: Tensor of shape (T,)
        phi: Scalar
        W: Tensor of shape (P, P)
        a: Scalar
        b: Scalar
        
    Returns:
        theta: Tensor of shape (T + 1, P)
    """
     
    T, P = F.shape
    device = Y.device
    dtype = Y.dtype
    
    theta = torch.zeros(T + 1, P, dtype = dtype, device = device)
    
    # filtered mean
    m = torch.zeros(T + 1, P, dtype = dtype, device = device) 
    
    # filtered covariance
    C = torch.zeros(T + 1, P, P, dtype = dtype, device = device)
    C[0,:,:] = torch.eye(P, dtype = dtype, device = device)
    
    R = torch.zeros(T, P, P, dtype = dtype, device = device)
    A = torch.zeros(T, P,  dtype = dtype, device = device)
    
    # Forward filtering
    for t in range(T):
        
        A[t, ] = G @ m[t, ]
        R[t, :, :] = G @ C[t, :, :] @ G.T + W
        
        RF = R[t, :, :] @ F[t : t + 1, ].T
        
        Qt = (F[t : t + 1,:] @ RF + b * U[t] / phi).squeeze()
        
        
        m[t + 1, ] = A[t, ] + RF.ravel() * ((Y[t] - F[t, ] @ A[t, ] - a * U[t]) / Qt)
        C[t + 1,:,:] = R[t,:,: ] - RF @ RF.T / Qt   
    
    theta[T,:] = MultivariateNormal(m[T, ], C[T,:,:]).sample()
    
    GCT = torch.einsum('tij,jk->tik', C[0 : T, :, :], G.T) 
        
    Cov = C[0 : T, :, :] - einsum('tij,tjk->tik', GCT, torch.linalg.solve(R, GCT.mT))
    
    R_inv_GC = solve(R, GCT.mT)
    
    theta[0 : T, ] = batch_mvn_sampler(Cov)
    
    # Backward sampling
    for t in range(T - 1, -1, -1):
        
        theta[t,:] +=  m[t, ] 
        theta[t,:] +=  R_inv_GC[t,:,:].T @ (theta[t + 1, ] - A[t, ])
  
    return theta


def Sample_AL(m, phi, Q):
    
    """
    Generate samples from Asymmetric Laplace distribution using inverse CDF.
    
    Args:
        m: location parameter
        phi: scale parameter (>0)
        Q: percentile parameter (0<p<1)
    
    Returns:
        Samples from Asymmetric Laplace distribution
    """
    
    u = torch.rand_like(m)
    
    if u.dim() == 2:
        
        return torch.where(u <= Q, m + (1 / (1 - Q) / phi).reshape(-1,1) * torch.log(u / Q), m - (1 / Q / phi).reshape(-1,1) * torch.log((1 - u) / (1 - Q)))
    
    else:
        
        return torch.where(u <= Q, m + (1 / (1 - Q) / phi) * torch.log(u / Q), m - (1 / Q / phi) * torch.log((1 - u) / (1 - Q)))
    

def NLL_AL(y, m, phi, Q):
    
    """
    Calculate log likelihood of Asymmetric Laplace Distribution.
    
    Args:
        y: Observed data
        m: Location parameter
        phi: Scale parameter (phi > 0)
        Q: Percentile parameter (0 < Q < 1)
    """
    
    diff = y - m.squeeze(-1)
    
    phi = phi.view(-1,1) 
    
    return -((Q * (1 - Q) * phi).log() - torch.where(diff <= 0, ((Q - 1) * phi) * diff, (Q * phi) * diff)).sum(-1)