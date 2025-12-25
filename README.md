# Time Series Simulation Toolbox

這是一個基於 Python 的時間序列模擬工具箱，結合了 **卡爾曼濾波 (Kalman Filter)** 的趨勢提取技術與多樣化的 **變點偵測 (Change Point Detection) Bootstrap** 。本工具旨在生成具備歷史統計特性與多維度相關性的合成數據，適用於任一高維度時間序列資料生成。

## 快速上手 (Quick Start)

確保環境已安裝：`numpy`, `pandas`, `matplotlib`, `yfinance`, `pykalman`, `hmmlearn`, `ruptures`。

```python
import yfinance as yf
from 模擬 import TimeSeriesSimulator, plot_results

# 1. 取得數據
data = yf.download('2330.TW', start='2020-01-01')[['Close', 'Volume']]

# 2. 初始化模擬器 (選擇 PELT 變點偵測方法，搭配 BIC 懲罰項)
sim_fixed = TimeSeriesSimulator(method='fixed', block_size=30)

# 3. 擬合數據並生成模擬路徑
sim_fixed.fit(data)

# 4. 可視化展示對比
plot_results(sim_fixed, data, sim_fixed.simulate(n=1)[0], "Fixed Block")
```

---

## 可視化輸出說明 (Visualization)
### 1. 數值對比圖 (Value Analysis)

* **藍色散點**：歷史觀測值
* **紅色實線**：Kalman 趨勢
* **橘色虛線**：模擬路徑
* **黑色垂直線**：變點位置

### 2. 殘差分析圖 (Residuals Analysis)

* 檢查模擬殘差的波動率、偏態是否與歷史一致

<img src="img/bspline.png" alt="B-spline basis" width="800">

---
## Block Bootstrap

### 1. 數據分解 (Kalman Decomposition)

對於多維時間序列觀測向量  
$$
\mathbf{y}_t \in \mathbb{R}^d
$$

透過卡爾曼濾波器（Kalman Filter）將其分解為趨勢與殘差：

$$
\mathbf{y}_t = \mathbf{T}_t + \boldsymbol{\epsilon}_t
$$

其中：

- $\mathbf{T}_t$：平滑後的潛在趨勢（state estimate）
- $\boldsymbol{\epsilon}_t$：去趨勢後的隨機殘差向量

為避免尺度差異影響後續建模，殘差進一步標準化：

$$
\tilde{\boldsymbol{\epsilon}}_t
= \frac{\boldsymbol{\epsilon}_t - \mu}{\sigma}
$$

---

### 2. 變點偵測與區段劃分 (Change Point Segmentation)

利用 `ruptures` 套件對標準化殘差矩陣  
$$
\tilde{\boldsymbol{\epsilon}}
= [\tilde{\boldsymbol{\epsilon}}_1, \ldots, \tilde{\boldsymbol{\epsilon}}_T]
$$
進行變點偵測，將時間軸切分為 $K$ 個統計上相對穩定的區段：

$$
\mathcal{S}_i
= \{ \tilde{\boldsymbol{\epsilon}}_t \mid t \in [\tau_i, \tau_{i+1}) \},
\quad i = 1, \ldots, K
$$

每個區段代表一種市場狀態（regime），例如：

- 高 / 低波動期
- 成交量結構轉換
- 價量關係改變

---

### 3. 區塊自助法（Block Bootstrap）

為了保留 **時間相依性（serial dependence）**，本工具在每個區段內採用 **Block Bootstrap** 生成殘差樣本。

#### 3.1 區塊定義

對於區段 $\mathcal{S}_i$，將其劃分為長度為 $b$ 的連續區塊：

$$
\mathcal{B}_{i,j}
= \{ \tilde{\boldsymbol{\epsilon}}_t \mid t \in [s_j, s_j + b) \}
$$

其中 $b$ 可為：

- 固定長度（Fixed Block Bootstrap）

---

#### 3.2 區塊重抽樣

在同一區段內，隨機（有放回）抽取區塊並串接，形成新的殘差序列：

$$
\tilde{\boldsymbol{\epsilon}}^{*(i)}
= \bigcup_{j=1}^{M_i} \mathcal{B}_{i, \pi_j}
$$

其中 $\pi_j$ 為隨機選取的區塊索引。

> **重要原則**：  
> Bootstrap 僅在「同一變點區段內」進行，避免跨 regime 混合導致結構失真。

---

### 4. 多變量共變異數重建 (Cross-Sectional Dependence)

為維持多維特徵（如 Close 與 Volume）之間的相關性，在每個區段 $\mathcal{S}_i$ 中估計共變異數矩陣：

$$
\Sigma_i
= \text{Cov}(\mathcal{S}_i)
$$

進行 Cholesky 分解：

$$
\Sigma_i = L_i L_i^\top
$$

---

### 5. 合成殘差生成 (Synthetic Residual Generation)

令標準常態隨機向量：

$$
\mathbf{Z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

則在區段 $i$ 中生成合成殘差：

$$
\hat{\boldsymbol{\epsilon}}_t
= \mu_i + L_i \mathbf{Z}_t
$$

此步驟確保：

- 區段內的變異結構一致
- 多變量橫斷面相關性被保留

---

### 6. 序列還原 (Reconstruction)

最後將合成殘差反標準化，並加回卡爾曼趨勢：

$$
\hat{\mathbf{y}}_t
= \mathbf{T}_t
+ \left(
\hat{\boldsymbol{\epsilon}}_t
\odot \sigma_{\text{original}}
\right)
$$

得到最終的模擬時間序列：

$$
\{ \hat{\mathbf{y}}_t \}_{t=1}^T
$$

---
## 支援模擬方法 (Supported Methods)

| 方法參數 (`method`) | 描述 (Algorithm Logic)  | 適用場景                
| --------------- | --------------------- | -------------------
| `hmm`           | Hidden Markov Model   | 狀態切換
| `pelt`          | PELT                  | 變點偵測
| `dynp`          | Dynamic Programming   | 已知變點數量
| `binseg`        | Binary Segmentation   | 大規模快速搜尋
| `kernel`        | Kernel Change Point   | 非線性分佈
| `window`        | Sliding Window        | 局部波動率變化
| `bottomup`      | Bottom-up Merge       | 微小變點合併 
| `fixed`         | Fixed Block Bootstrap | 傳統區塊模擬

# Time-Series-Simulation-Toolbox
