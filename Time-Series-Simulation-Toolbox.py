#%% [1] 匯入模組
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pykalman import KalmanFilter
from hmmlearn.hmm import GaussianHMM
import ruptures as rpt

#%% [2] 提取趨勢：Kalman Filter
class BaseSimulator:
    def __init__(self, n_iter_kf=10):
        self.n_iter_kf = n_iter_kf
        self.stats = {}
        self.trends = []      
        self.residuals = []   
        self.columns = []
        self.data_index = None

    def _extract_trends(self, df):
        self.data_index = df.index
        self.columns = df.columns.tolist()
        all_std_residuals = []
        self.trends = []
        for col in self.columns:
            data_vec = df[col].values
            f_mean, f_std = data_vec.mean(), data_vec.std()
            self.stats[col] = (f_mean, f_std)
            std_data = (data_vec - f_mean) / (f_std + 1e-9)
            
            # 使用 Kalman Filter 提取趨勢 (Trend Extraction)
            kf = KalmanFilter(initial_state_mean=std_data[0], n_dim_obs=1)
            kf = kf.em(std_data, n_iter=self.n_iter_kf)
            state_means, _ = kf.filter(std_data)
            
            trend_std = state_means.flatten()
            self.trends.append(trend_std * f_std + f_mean)
            res_std = std_data - trend_std
            all_std_residuals.append(res_std)
        return np.column_stack(all_std_residuals)

#%% [3] 模擬引擎 A: HMM (隱馬可夫模型)
class HMMEngine(BaseSimulator):
    def __init__(self, n_states=3, n_iter_kf=10):
        super().__init__(n_iter_kf)
        self.n_states = n_states

    def fit(self, df):
        res_matrix = self._extract_trends(df)
        self.hmm = GaussianHMM(n_components=self.n_states, covariance_type='full', n_iter=100)
        self.hmm.fit(res_matrix)

    def simulate(self, n=1):
        results = []
        for _ in range(n):
            res, _ = self.hmm.sample(len(self.data_index))
            df = pd.DataFrame(index=self.data_index, columns=self.columns)
            for i, c in enumerate(self.columns):
                df[c] = self.trends[i] + res[:, i] * self.stats[c][1]
            results.append(df)
        return results

#%% [4] 模擬引擎 B: Bootstrap (整合 Ruptures 所有搜尋演算法)
class BootstrapEngine(BaseSimulator):
    def __init__(self, block_method='pelt', model='l2', kernel='rbf', 
                 n_bkps=5, penalty=None, width=10, block_size=20, n_iter_kf=10):
        super().__init__(n_iter_kf)
        self.block_method = block_method
        self.model = model
        self.kernel = kernel
        self.n_bkps = n_bkps
        self.penalty = penalty
        self.width = width
        self.block_size = block_size
        self.segments = []

    def fit(self, df):
        res_matrix = self._extract_trends(df)
        n_samples, n_features = res_matrix.shape

        if self.block_method == 'fixed':
            n_blocks = n_samples // self.block_size
            self.segments = [(i * self.block_size, (i + 1) * self.block_size) for i in range(n_blocks)]
            if n_samples % self.block_size != 0:
                self.segments.append((n_blocks * self.block_size, n_samples))
        else:
            # 依照 Ruptures 指南選擇搜尋方法
            if self.block_method == 'dynp':
                algo = rpt.Dynp(model=self.model).fit(res_matrix)
            elif self.block_method == 'pelt':
                algo = rpt.Pelt(model=self.model).fit(res_matrix)
            elif self.block_method == 'kernel':
                algo = rpt.KernelCPD(kernel=self.kernel).fit(res_matrix)
            elif self.block_method == 'binseg':
                algo = rpt.Binseg(model=self.model).fit(res_matrix)
            elif self.block_method == 'bottomup':
                algo = rpt.BottomUp(model=self.model).fit(res_matrix)
            elif self.block_method == 'window':
                algo = rpt.Window(width=self.width, model=self.model).fit(res_matrix)
            else:
                raise ValueError(f"不支援的方法: {self.block_method}")

            # 決定變點預測方式
            if self.penalty is not None:
                if self.penalty == 'bic':
                    actual_pen = n_features * np.log(n_samples)
                    result = algo.predict(pen=actual_pen)
                else:
                    result = algo.predict(pen=self.penalty)
            else:
                result = algo.predict(n_bkps=self.n_bkps)

            indices = [0] + result
            self.segments = [(indices[i], indices[i+1]) for i in range(len(indices)-1)]
            print(f"變點偵測 [{self.block_method}] 完成：{len(self.segments)} 個區塊")
            
        self.res_matrix = res_matrix

    def simulate(self, n_simulations=1):
        n_samples, n_features = self.res_matrix.shape
        results = []
        for _ in range(n_simulations):
            pseudo_res = np.zeros_like(self.res_matrix)
            for start, end in self.segments:
                block_data = self.res_matrix[start:end, :]
                block_len = end - start
                if block_len < 2:
                    pseudo_res[start:end, :] = block_data
                    continue
                cov = np.cov(block_data, rowvar=False) + np.eye(n_features) * 1e-9
                L = np.linalg.cholesky(cov)
                noise = (L @ np.random.normal(0, 1, (n_features, block_len))).T
                pseudo_res[start:end, :] = noise + block_data.mean(axis=0)
            
            sim_df = pd.DataFrame(index=self.data_index, columns=self.columns)
            for i, col in enumerate(self.columns):
                sim_df[col] = self.trends[i] + (pseudo_res[:, i] * self.stats[col][1])
            results.append(sim_df)
        return results

#%% [5] 統一介面：模擬器工廠
class TimeSeriesSimulator:
    def __init__(self, method='hmm', **kwargs):
        if method == 'hmm':
            self.engine = HMMEngine(**kwargs)
        else:
            self.engine = BootstrapEngine(block_method=method, **kwargs)
    
    def fit(self, df): self.engine.fit(df)
    def simulate(self, n=1): return self.engine.simulate(n)
    def get_engine(self): return self.engine

#%% [6] 繪圖工具函數
def plot_results(sim, original_df, sim_df, title="Result"):
    engine = sim.get_engine()
    
    # 為每一個欄位創建一個獨立的 Figure
    for i, col in enumerate(original_df.columns):
        # 建立 2x1 的子圖佈局
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # --- (a) 原始數據與濾波後數據 ---
        # 使用散點圖繪製原始數據 (scatter)
        axes[0].scatter(original_df.index, original_df[col], 
                        label='data', color='blue', alpha=0.3, s=10)
        
        # 繪製趨勢線 (Trend/Filtered)
        axes[0].plot(original_df.index, engine.trends[i], 
                     label='filtered', color='red', linewidth=2)
        
        # 繪製模擬線 (Simulated)
        axes[0].plot(sim_df.index, sim_df[col], 
                     label='simulated', color='orange', linestyle='--', alpha=0.8)
        
        # 如果有分段資訊，繪製垂直線
        if hasattr(engine, 'segments'):
            for s, _ in engine.segments:
                idx = min(s, len(original_df) - 1)
                axes[0].axvline(original_df.index[idx], color='black', alpha=0.2)
                axes[1].axvline(original_df.index[idx], color='black', alpha=0.2)
        
        axes[0].set_title(f"{title} - {col} (Price/Value)")
        axes[0].set_ylabel("Value")
        axes[0].legend()

        # --- (b) 殘差 (Residuals) ---
        # 計算殘差：原始數據 - 趨勢線
        residuals = original_df[col].values - engine.trends[i]
        
        axes[1].plot(original_df.index, residuals, label='residuals', color='green')
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
        
        axes[1].set_title(f"{col} - Residuals")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Residuals")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

#%% [7] 下載數據
data = yf.download('2330.TW', start='2020-01-01')[['Close', 'Volume']]

#%% [8] 測試 1：Fixed Block Bootstrap
print("\n>>> 執行固定大小 Bootstrap:")
sim_fixed = TimeSeriesSimulator(method='fixed', block_size=30)
sim_fixed.fit(data)
plot_results(sim_fixed, data, sim_fixed.simulate(1)[0], "Fixed Block")

#%% [9] 測試 2：PELT
print("\n>>> 執行 PELT + BIC 自動懲罰:")
sim_pelt = TimeSeriesSimulator(method='pelt', model='l2', penalty='bic')
sim_pelt.fit(data)
plot_results(sim_pelt, data, sim_pelt.simulate(1)[0], "PELT")

#%% [10] 測試 3：Binary Segmentation
print("\n>>> 執行 Binary Segmentation:")
sim_binseg = TimeSeriesSimulator(method='binseg', model='l2', n_bkps=5)
sim_binseg.fit(data)
plot_results(sim_binseg, data, sim_binseg.simulate(1)[0], "BinSeg")

#%% [11] 測試 4：Dynamic Programming (Dynp)
print("\n>>> 執行 Dynamic Programming (Dynp):")
sim_dynp = TimeSeriesSimulator(method='dynp', model='l2', n_bkps=4)
sim_dynp.fit(data)
plot_results(sim_dynp, data, sim_dynp.simulate(1)[0], "Dynamic Programming")

#%% [12] 測試 5：Kernel Change Detection (KernelCPD)
print("\n>>> 執行 Kernel Change Detection (KernelCPD):")
sim_kernel = TimeSeriesSimulator(method='kernel', kernel='rbf', n_bkps=4)
sim_kernel.fit(data)
plot_results(sim_kernel, data, sim_kernel.simulate(1)[0], "Kernel CPD")

#%% [13] 測試 6：Bottom-up Segmentation
print("\n>>> 執行 Bottom-up Segmentation:")
sim_bottomup = TimeSeriesSimulator(method='bottomup', model='l2', n_bkps=5)
sim_bottomup.fit(data)
plot_results(sim_bottomup, data, sim_bottomup.simulate(1)[0], "Bottom-up")

#%% [14] 測試 7：Window Sliding Segmentation
print("\n>>> 執行 Window Sliding Segmentation (Search Method Documentation):")
sim_window = TimeSeriesSimulator(method='window', width=20, model='l2', penalty=10)
sim_window.fit(data)
plot_results(sim_window, data, sim_window.simulate(1)[0], "Window Sliding")

#%% [15] 測試 8：使用 HMM 方法
print("\n>>> 使用 HMM 方法進行模擬:")
sim_hmm = TimeSeriesSimulator(method='hmm', n_states=1)
sim_hmm.fit(data)
plot_results(sim_hmm, data, sim_hmm.simulate(1)[0], title="HMM Method")