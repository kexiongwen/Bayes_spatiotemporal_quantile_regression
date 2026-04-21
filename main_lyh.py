#!/usr/bin/env python
# coding: utf-8

# In[1]:
import  pandas as pd
import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
from Bayes_ST_QR.BSTQR import Spatiotemporal_Bayes_Quantile_regression
#from temperature_forecast_script_30_days import y_mean

# In[2]:

#file_path = r"D:\教学事务\学生科研\2022级本科生毕业论文\源代码\Bayes_spatiotemporal_quantile_regression-main\station_500.nc"
ds = pd.read_csv('s_500.csv')
df = ds.values.flatten()
y = df[0:165]


# 检查并填充缺失值
if np.any(np.isnan(y)):
    print("数据中存在 NaN，将使用前向填充处理")
    y = pd.Series(y).ffill().bfill().values



N = len(y)  # 3 years daily
t = np.arange(N)

# Simulated temperature (seasonal + noise)
# y = 20+ np.random.normal(0, 2, N)
#y = 20 + 10*np.sin(2*np.pi*t/365) + np.random.normal(0, 2, N)


# In[3]:


# STANDARDIZE
y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std

# -----------------------------
# BUILD MODEL MATRICES
# -----------------------------
FF = np.ones((3,N))  #for no seasonality
#FF = np.vstack([
#    np.ones(N),
#    np.sin(2 * np.pi * t / 365),
#    np.cos(2 * np.pi * t / 365)
#])

GG = np.eye(3)


# In[4]:


# device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device1 = torch.device("xpu" if torch.xpu.is_available() else "cpu")
# print(f'using:{device1}')


# In[5]:


device1 = "cpu"


# In[6]:


model = Spatiotemporal_Bayes_Quantile_regression(Y = y_scaled, F = FF.T, G = GG, Q = 0.5, device = device1)
model_upp = Spatiotemporal_Bayes_Quantile_regression(Y = y_scaled, F = FF.T, G = GG, Q = 0.95, device = device1)
model_lower = Spatiotemporal_Bayes_Quantile_regression(Y = y_scaled, F = FF.T, G = GG, Q = 0.05, device = device1)
model_99 = Spatiotemporal_Bayes_Quantile_regression(Y=y_scaled, F=FF.T, G=GG, Q=0.99, device=device1)

# In[7]:


model.Gibbs_sampler()


# In[8]:


model_upp.Gibbs_sampler()


# In[9]:


model_lower.Gibbs_sampler()
model_99.Gibbs_sampler()

# In[10]:


model.fit_Y()
model_upp.fit_Y()
model_lower.fit_Y()
model_99.fit_Y()

# In[11]:


#future_true = site_data.sel(time=slice('2021-07-01', '2021-07-30')).values
future_true = df[166:181]



# FORECAST 30 DAYS
H = 15
F_future = np.zeros((3, H))
if len(future_true) < H:
    print(f"警告：未来真实数据只有 {len(future_true)} 天，预测 {H} 天，将用 NaN 填充缺失部分。")
    future_true = np.pad(future_true, (0, H - len(future_true)), constant_values=np.nan)

for h in range(H):
    F_future[:, h] = np.array([1, 0, 0])  # 第一维为1，其余为0（与 FF = np.ones((3,N)) 对应）


# In[12]:


Y_pred = model.predict_Y(F_future)
Y_pred_upp = model_upp.predict_Y(F_future)
Y_pred_lower = model_lower.predict_Y(F_future)
Y_pred_99 = model_99.predict_Y(F_future)

# In[13]:


past_pred = model.y_fitted.to('cpu').numpy() * y_std + y_mean
past_pred_upp = model_upp.y_fitted.to('cpu').numpy() * y_std + y_mean
past_pred_lower = model_lower.y_fitted.to('cpu').numpy() * y_std + y_mean
past_pred_99 = model_99.y_fitted.to('cpu').numpy() * y_std + y_mean

# In[14]:


Y_pred = Y_pred.to('cpu').numpy() * y_std + y_mean
Y_pred_upp = Y_pred_upp.to('cpu').numpy() * y_std + y_mean
Y_pred_lower = Y_pred_lower.to('cpu').numpy() * y_std + y_mean
Y_pred_99 = Y_pred_99.to('cpu').numpy() * y_std + y_mean

# In[21]:


#visualize
plt.figure(figsize=(12,5))
plt.plot(np.arange(N, N+H), future_true, 's-', label="True Future", color='blue', markersize=6, linewidth=2)
plt.plot(np.arange(N), y, label="Observations", linewidth=1)
plt.plot(np.arange(N+H), np.concatenate([past_pred.mean(0), Y_pred.mean(0)]), label="Median", linewidth=1, color = 'orange')
plt.plot(np.arange(N+H), np.concatenate([past_pred_upp.mean(0), Y_pred_upp.mean(0)]), label="95% Quantile", linestyle='--', color = 'red')
plt.plot(np.arange(N+H), np.concatenate([past_pred_lower.mean(0), Y_pred_lower.mean(0)]), label="5% Quantile", linestyle='--', color = 'green')
plt.plot(np.arange(N+H), np.concatenate([past_pred_99.mean(0), Y_pred_99.mean(0)]),
         label="99% Quantile", linestyle='--', color='purple')
plt.axvline(N, linestyle=':')
plt.legend()
plt.show()

