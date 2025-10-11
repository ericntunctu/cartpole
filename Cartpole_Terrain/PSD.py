import os, zipfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# === 建立 recorded_data ===
recorded_data = {
    'cart_positions': np.array(cart_positions),
    'cart_velocities': np.array(cart_velocities),
    'pole_angles': np.array(pole_angles),
    'pole_angular_velocities': np.array(pole_angular_velocities),
    'actions_taken': np.array(actions_taken),
    'rewards_received': np.array(rewards_received),
    'timestamps': np.array(timestamps),
    'total_steps': step_count,
    'total_reward': total_reward,
    'pole_tip_offsets': np.array(pole_tip_offsets),
    'episode_rewards': np.array([]),
    'phis': np.array(phis),

    # === 記錄正在訓練的地形 ===
    'terrain_name': np.array([TERRAIN_NAME]),
    'terrain_parameter': np.array([TERRAIN_PARAM], dtype=object)
}

# === 儲存 .npz 到原始輸出資料夾 ===
data_filename = f"Terrain_{TERRAIN_TAG}_PoleLength_{length}_PoleMass_{mass}_Friction_{friction_coef}.npz"
npz_path = os.path.join(output_dir, data_filename)
np.savez(npz_path, **recorded_data)

# === 壓縮成 .zip 丟到 OneDrive ===
target_dir = r"C:\Users\kyleh\OneDrive\文件\HRV & CartPole\Cartpole_Terrain_Code\Sine_Terrain_Data"
os.makedirs(target_dir, exist_ok=True)
zip_filename = os.path.splitext(data_filename)[0] + ".zip"
zip_path = os.path.join(target_dir, zip_filename)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(npz_path, os.path.basename(npz_path))

# === 自動呼叫數據分析 ===
def analyze_recorded_data(data_path):
    """依據你的分析邏輯產生三張圖並儲存"""
    data = np.load(data_path, allow_pickle=True)
    pole_tip_offsets = data["pole_tip_offsets"]
    timestamps = data["timestamps"]

    zero_crossing_indices = np.where(np.diff(np.sign(pole_tip_offsets)) != 0)[0]
    zero_crossing_times = timestamps[zero_crossing_indices + 1]

    # ---------- 圖 1：桿端振盪與零交越 ----------
    plt.figure(figsize=(9, 5))
    plt.plot(timestamps[:500], pole_tip_offsets[:500], label='Pole Tip Offset')
    plt.scatter(zero_crossing_times[:500], np.zeros_like(zero_crossing_times)[:500],
                color='red', label='Zero Crossings')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Pole Tip Horizontal Offset")
    plt.legend(); plt.title("Pole Tip Oscillation and Zero Crossings")
    plt.grid(True); plt.tight_layout()
    fig1_path = os.path.join(target_dir, os.path.splitext(os.path.basename(data_path))[0] + "_plot1.png")
    plt.savefig(fig1_path, dpi=150)
    plt.close()

    # ---------- 圖 2：零交越週期頻譜 ----------
    even_times = zero_crossing_times[::2]
    periods = np.diff(even_times)
    N = len(periods)
    dt = 1
    yf = fft(periods)
    xf = fftfreq(N, dt)
    xf_pos = xf[:N//2]
    power_spectrum = np.abs(yf[:N//2])**2

    plt.figure(figsize=(10, 5))
    plt.plot(xf_pos, power_spectrum)
    plt.xlabel("Frequency (1/crossings)")
    plt.ylabel("Power")
    plt.title("Fourier Spectrum of Zero-Crossing Periods")
    plt.grid(True)
    fig2_path = os.path.join(target_dir, os.path.splitext(os.path.basename(data_path))[0] + "_plot2.png")
    plt.savefig(fig2_path, dpi=150)
    plt.close()

    # ---------- 圖 3：log–log 頻譜 ----------
    periods = periods - np.mean(periods)
    N = len(periods)
    yf = fft(periods)
    xf = fftfreq(N, dt)
    xf = xf[:N//2]
    power = np.abs(yf[:N//2])**2
    eps = 1e-12
    log_xf = np.log10(xf[2:] + eps)
    log_power = np.log10(power[2:] + eps)
    plt.figure(figsize=(8, 5))
    plt.plot(log_xf, log_power, label='Power Spectrum (log-log)')
    plt.xlabel('log10(Frequency)')
    plt.ylabel('log10(Power)')
    plt.title('Low-frequency behavior of Power Spectrum')
    plt.grid(True); plt.legend(); plt.tight_layout()
    fig3_path = os.path.join(target_dir, os.path.splitext(os.path.basename(data_path))[0] + "_plot3.png")
    plt.savefig(fig3_path, dpi=150)
    plt.close()

    print(f"✅ 三張圖已儲存：\n  {fig1_path}\n  {fig2_path}\n  {fig3_path}")

# 呼叫分析
analyze_recorded_data(npz_path)

# === 顯示進度 ===
if tqdm:
    tqdm.write(f"Saved data: {npz_path}")
    tqdm.write(f"Saved zip:  {zip_path}")
    outer_bar.update(1)
else:
    print(f"Saved data: {npz_path} ({combo_idx}/{grid_total})")
    print(f"Saved zip:  {zip_path}")
