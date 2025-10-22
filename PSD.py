INPUT_NPZ  = r"C:\Users\kyleh\OneDrive\文件\HRV & CartPole\Cartpole_Terrain_Code\Cartpole_Terrain\Code\cartpole_data\Terrain_sin_A0.02_k0.02_tx3.0_PoleLength_1.0_PoleMass_1.0_Friction_0.npz"
OUTPUT_DIR = r"C:\Users\kyleh\OneDrive\文件\HRV & CartPole\Cartpole_Terrain_Code\Sine_Terrain_Data"       # 圖片輸出資料夾

# ===== 分析程式（產生三張圖）=====
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

os.makedirs(OUTPUT_DIR, exist_ok=True)
base = os.path.splitext(os.path.basename(INPUT_NPZ))[0]

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
phis = data["phis"]

# 額外標題資訊（若存在）
terrain_name  = data["terrain_name"][0]        if "terrain_name" in data.files else ""
terrain_param = data["terrain_parameter"][0]   if "terrain_parameter" in data.files else {}

# ---------- 圖1：桿端位移 + 零交越 ----------
zero_idx = np.where(np.diff(np.sign(pole_tip_offsets)) != 0)[0]
zero_times = timestamps[zero_idx + 1]

plt.figure(figsize=(9, 5))
plt.plot(timestamps[:500], pole_tip_offsets[:500], label='Pole Tip Offset')
plt.scatter(zero_times[:500], np.zeros_like(zero_times)[:500], color='red', s=14, label='Zero Crossings')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Pole Tip Horizontal Offset")
ttl1 = f"Zero Crossings | {terrain_name} {terrain_param}" if len(str(terrain_name))>0 else "Zero Crossings"
plt.title(ttl1)
plt.legend(); plt.grid(True); plt.tight_layout()
fig1 = os.path.join(OUTPUT_DIR, f"{base}_plot1_zero_crossings.png")
plt.savefig(fig1, dpi=150); plt.show()

# ---------- 圖2：零交越 periods 的傅立葉頻譜 ----------
even_times = zero_times[::2]
periods = np.diff(even_times)            # RR-like
N = len(periods)
dt = 1.0                                 # 以索引當等間距
yf = fft(periods)                        # 依你原版，不先去均值
xf = fftfreq(N, dt)
xf_pos = xf[:N//2]
power_spectrum = np.abs(yf[:N//2])**2

plt.figure(figsize=(10, 5))
plt.plot(xf_pos, power_spectrum)
plt.xlabel("Frequency (1/crossings)")
plt.ylabel("Power")
plt.title("Fourier Spectrum of Zero-Crossing Periods")
plt.grid(True); plt.tight_layout()
fig2 = os.path.join(OUTPUT_DIR, f"{base}_plot2_period_spectrum.png")
plt.savefig(fig2, dpi=150); plt.show()

# ---------- 圖3：頻譜低頻 log–log ----------
# ---------- 圖3：頻譜低頻 log–log ----------  
periods = periods - np.mean(periods)     # 去均值再看趨勢  
N = len(periods)  
yf = fft(periods)  
xf = fftfreq(N, dt)  
xf = xf[:N//2]  
power = np.abs(yf[:N//2])**2  
eps = 1e-12  
log_x = np.log10(xf[2:] + eps)           # 略過前兩個頻點避免 DC/極低頻  
log_p = np.log10(power[2:] + eps)  

plt.figure(figsize=(8, 5))
plt.scatter(log_x, log_p, s=15, color='blue', label='Power Spectrum (log-log)')  # 改成散點圖
plt.xlabel('log10(Frequency)')
plt.ylabel('log10(Power)')
plt.title('Low-frequency behavior of Power Spectrum')
plt.grid(True)
plt.legend()
plt.tight_layout()

fig3 = os.path.join(OUTPUT_DIR, f"{base}_plot3_loglog.png")
plt.savefig(fig3, dpi=150)
plt.show()


print("Saved:")
print(fig1)
print(fig2)
print(fig3)
