import os
import numpy as np
import matplotlib.pyplot as plt

# 設定路徑
INPUT_NPZ = r"C:\Users\kyleh\OneDrive\文件\HRV & CartPole\Cartpole_Terrain_Code\Cartpole_Terrain\Code\cartpole_data\Terrain_sin_A0.02_k0.02_tx3.0_PoleLength_1.0_PoleMass_1.0_Friction_0.npz"
OUTPUT_DIR = r"C:\Users\kyleh\OneDrive\文件\HRV & CartPole\Cartpole_Terrain_Code\Sine_Terrain_Data"  # 圖片輸出資料夾

os.makedirs(OUTPUT_DIR, exist_ok=True)
base = os.path.splitext(os.path.basename(INPUT_NPZ))[0]

# --------- 修正的 DFA 函式（移除有重疊的反向 loop，只用非重疊段；加邊界處理）---------
def DFA(X, n_range):
    size = len(X)
    μ = np.mean(X)
    Y = np.cumsum(X - μ)
    deg = 1
    F = []

    for n in n_range:
        if n > size:
            F.append(np.nan)
            continue
        F2 = []
        # 只用非重疊段，從頭開始
        for start in range(0, size - n + 1, n):  # +1 確保包括最後可能段；step=n 非重疊
            try:
                Z = Y[start:start + n]
                if len(Z) < n:  # 如果最後段不足n，跳過（避免偏誤）
                    continue
                X_fit = np.arange(n)
                coeff = np.polyfit(X_fit, Z, deg)
                Z_fit = np.polyval(coeff, X_fit)
                F2.append(np.mean((Z - Z_fit) ** 2))
            except ValueError:
                # polyfit 失敗（e.g., n < deg+1），跳過此段
                continue

        if len(F2) > 0:
            F.append(np.sqrt(np.mean(F2)))
        else:
            F.append(np.nan)

    F = np.array(F)
    mask = np.isfinite(F)  # 濾除 NaN/Inf
    F = F[mask]
    n_range_filtered = np.array([n for n, f in zip(n_range, F) if np.isfinite(f)])

    if len(F) < 2:
        raise ValueError("DFA 計算後有效尺度點太少，無法擬合。")

    log_n = np.log(n_range_filtered)
    log_F = np.log(F)
    coeff = np.polyfit(log_n, log_F, 1)

    return F, n_range_filtered, coeff[0]

def make_n_range(N, min_n=4, max_frac=0.25, num=20):
    """
    幫 periods 序列產生合適的 n_range（對數等距的整數）。
    max_frac: 最大尺度佔序列長度的比例（預設 1/4）
    """
    if N < 20:
        # 序列太短，退而求其次
        vals = np.arange(4, max(5, N//2) + 1)
        return vals[vals > 1]
    max_n = max(8, int(N * max_frac))
    n_cont = np.logspace(np.log10(4), np.log10(max_n), num=num)
    n_vals = np.unique(np.clip(np.rint(n_cont).astype(int), 2, N-1))
    return n_vals

# 讀資料（你的 key）
data = np.load(INPUT_NPZ, allow_pickle=True)
cart_pos = data["cart_positions"]
cart_vel = data["cart_velocities"]
pole_ang = data["pole_angles"]
pole_ang_vel = data["pole_angular_velocities"]
actions = data["actions_taken"]
rewards = data["rewards_received"]
timestamps = data["timestamps"]
pole_tip_offsets = data["pole_tip_offsets"]
phis= data["phis"]

# 額外標題資訊（若存在）
terrain_name = data["terrain_name"][0] if "terrain_name" in data.files else ""
if "terrain_parameter" in data.files:
    terrain_param_raw = data["terrain_parameter"][0]
    terrain_param = str(terrain_param_raw) if not isinstance(terrain_param_raw, str) else terrain_param_raw
else:
    terrain_param = ""

# ---------- 圖1：桿端位移 + 零交越（修：動態取長度，避免 IndexError） ----------
zero_idx = np.where(np.diff(np.sign(pole_tip_offsets)) != 0)[0]
zero_times = timestamps[zero_idx + 1]  # 交越後一點的時間作為近似

plot_len = min(500, len(pole_tip_offsets), len(timestamps))
plt.figure(figsize=(9, 5))
plt.plot(timestamps[:plot_len], pole_tip_offsets[:plot_len], label='Pole Tip Offset')
scatter_len = min(plot_len, len(zero_times))
plt.scatter(zero_times[:scatter_len], np.zeros(scatter_len), color='red', s=14, label='Zero Crossings')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Pole Tip Horizontal Offset")
ttl1 = f"Zero Crossings | {terrain_name} {terrain_param}" if terrain_name else "Zero Crossings"
plt.title(ttl1)
plt.legend(); plt.grid(True); plt.tight_layout()
fig1 = os.path.join(OUTPUT_DIR, f"{base}_plot1_zero_crossings.png")
plt.savefig(fig1, dpi=150); plt.show()

# ---------- 建立「零交越 periods」序列 ----------
# 取偶數 crossing 當週期端點（避免半週）
even_times = zero_times[::2]
periods = np.diff(even_times)  # RR-like，單位：秒
periods = periods[np.isfinite(periods)]
N = len(periods)

if N < 10:
    raise ValueError(f"零交越 periods 太少（N={N}），無法做穩定的 DFA。請更長的資料或放寬條件。")

# ---------- 圖2：DFA（log F 對 log n） ----------
n_range = make_n_range(N, min_n=4, max_frac=0.25, num=24)
F, ns, alpha = DFA(periods, n_range)

log_n = np.log(ns)
log_F = np.log(F)
fit = np.polyfit(log_n, log_F, 1)           # 斜率 ~ alpha
fit_line = np.polyval(fit, log_n)

plt.figure(figsize=(10, 5))
plt.plot(log_n, log_F, 'o', label='DFA F(n)')         # 散點顯示
plt.plot(log_n, fit_line, '-', label=f'fit: α={fit[0]:.3f}')
plt.xlabel("log n")
plt.ylabel("log F(n)")
ttl2 = f"DFA on Zero-Crossing Periods | {terrain_name} {terrain_param}" if terrain_name else "DFA on Zero-Crossing Periods"
plt.title(ttl2)
plt.grid(True); plt.legend(); plt.tight_layout()
fig2 = os.path.join(OUTPUT_DIR, f"{base}_plot2_DFA_loglog.png")
plt.savefig(fig2, dpi=150); plt.show()

# ---------- 圖3：局部斜率 α(n)（觀察 α 穩定度） ----------
# 使用相鄰點差分估局部斜率：d logF / d logn
if len(ns) >= 3:
    local_alpha = np.diff(log_F) / np.diff(log_n)
    ns_mid = np.sqrt(ns[1:] * ns[:-1])  # 幾何中點代表的尺度
    plt.figure(figsize=(8, 5))
    plt.plot(ns_mid, local_alpha, 'o-', label='local α(n)')
    plt.xscale('log')
    plt.xlabel('n (scale, log)')
    plt.ylabel('local slope α(n)')
    plt.axhline(alpha, color='gray', linestyle='--', label=f'global α={alpha:.3f}')
    plt.title('Scale-dependent α(n)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    fig3 = os.path.join(OUTPUT_DIR, f"{base}_plot3_DFA_local_alpha.png")
    plt.savefig(fig3, dpi=150); plt.show()
else:
    # 尺度太少無法畫局部斜率
    fig3 = os.path.join(OUTPUT_DIR, f"{base}_plot3_DFA_local_alpha.png")
    with open(fig3.replace(".png", ".txt"), "w", encoding="utf-8") as f:
        f.write("Not enough DFA scales to compute local alpha.\n")
    print("DFA 尺度點太少，略過局部斜率圖。")

print("Saved:")
print(fig1)
print(fig2)
print(fig3)