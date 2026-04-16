import torch
from tqdm import tqdm
from torch import einsum
import matplotlib.pyplot as plt
from torch.linalg import cholesky
from torch.distributions.gamma import Gamma
from Bayes_ST_QR.utils import inv_gauss, Sample_IW, FFBS, Sample_AL, NLL_AL


class Spatiotemporal_Bayes_Quantile_regression:

    def __init__(self, Y, F, G, Q, device, M = 2000, burn_in = 2000):
        
        
        self.dtype = torch.float64
        self.device = device
        
        self.Y = torch.from_numpy(Y).to(self.dtype).to(self.device)
        self.F = torch.from_numpy(F).to(self.dtype).to(self.device)
        self.G = torch.from_numpy(G).to(self.dtype).to(self.device)
        self.Q = Q
        
        self.M = M # number of MCMC iterations
        self.burn_in = burn_in # number of burn-in iterations
    
    def Gibbs_sampler(self):
        
        a = (1 - 2 * self.Q) / (self.Q * (1 - self.Q))
        b = 2 / (self.Q * (1 - self.Q))
        
        T, P = self.F.shape
        
        theta_samples = []
        phi_samples = []
        W_samples = []
    
        # initial state
        theta_sample = torch.randn(T + 1, P, dtype = self.dtype, device = self.device)
        phi_sample = torch.tensor(1.0, dtype = self.dtype, device = self.device)
       
       
        for i in tqdm(range(self.M + self.burn_in)):
        
            error = self.Y - (self.F * theta_sample[1:,]).sum(dim = 1)
            
            # Sample phi
            phi_sample = Gamma(1 + T, torch.maximum(self.Q * error, (self.Q - 1) * error).sum() + 1).sample()
            
            # Sample U
            U =  (2 * self.Q * (1 - self.Q)) / phi_sample  / inv_gauss(2 / phi_sample / torch.maximum(error.abs().ravel(),torch.tensor(1e-6, device = self.device, dtype = self.dtype))) 
            
            # Sample W
            res = theta_sample[1:,].T - self.G @ theta_sample[0:-1,].T
            W_sample = Sample_IW(res @ res.T + torch.eye(P, device = self.device, dtype = self.dtype), T + P)
            
            # Sample theta
            theta_sample = FFBS(self.Y, self.F, self.G, U, phi_sample, W_sample, a, b)
            
            if i >= self.burn_in:
                
                theta_samples.append(theta_sample)
                phi_samples.append(phi_sample)
                W_samples.append(W_sample)
                
        
        self.results = {"theta_samples": torch.stack(theta_samples), "phi_samples": torch.stack(phi_samples), "W_samples": torch.stack(W_samples)}
        
        NLL = NLL_AL(self.Y, (self.F * self.results["theta_samples"][:,1:,:]).sum(-1), self.results["phi_samples"], Q = self.Q)
        plt.figure()
        plt.plot(NLL.cpu().numpy())
        plt.title("Trace plot")
        plt.ylabel("Negative Log-Likelihood")
        plt.xlabel("Iteration")
        plt.show(block=False)
        
    
    def fit_Y(self):
        
        theta_samples = self.results["theta_samples"]
        self.y_fitted = Sample_AL((theta_samples[:,1:,:] * self.F).sum(-1), self.results["phi_samples"], self.Q)
        
    
    def predict_Y(self, F_new):
        
        F_new = torch.from_numpy(F_new).to(self.dtype).to(self.device)
        H = F_new.shape[1]
        
        theta_last = self.results["theta_samples"][:,-1,:]
        W_samples = self.results["W_samples"]
        
        N, P, _ = W_samples.shape
             
        # Generate standard normal samples
        z = torch.randn(N, P, 1, dtype = self.dtype, device = self.device)
        
        # Transform standard normal samples to match the covariance structure W_samples
        z = torch.bmm(cholesky(W_samples), z).squeeze(-1)
        
        # Initialize the predicted values tensor
        Y_pred = torch.zeros(N, H, dtype = self.dtype, device = self.device)
        
        # First time step prediction
        theta_t = torch.einsum('ij,tj->ti', self.G, theta_last) + z[0,:]
        Y_pred[:, 0] = Sample_AL((theta_t * F_new[:, 0]).sum(-1), self.results["phi_samples"], self.Q)
        
        # Subsequent time steps prediction
        for i in range(1, H):
            
            theta_t = torch.einsum('ij,tj->ti', self.G, theta_t) + z[i,:]
            Y_pred[:, i] = Sample_AL((theta_t * F_new[:,i]).sum(-1), self.results["phi_samples"], self.Q)
                
        return Y_pred
    


    
        
        
        
        
        
    
        
        
        
        
        