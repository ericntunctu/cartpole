import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq



def fourier_analysis(path):
    data = np.load(path)

    cart_pos = data["cart_positions"]
    cart_vel = data["cart_velocities"]
    pole_ang = data["pole_angles"]
    pole_ang_vel = data["pole_angular_velocities"]
    pole_tip_offsets = data["pole_tip_offsets"]
    timestamps = data["timestamps"]
    
    zero_crossing_indices = np.where(np.diff(np.sign(pole_tip_offsets)) != 0)[0]
    
    # 加上 1，因為 np.diff 產生的是 N-1 長度
    zero_crossing_times = timestamps[zero_crossing_indices + 1]
    plt.plot(timestamps[:500], pole_tip_offsets[:500], label='Pole Tip Offset')
    plt.scatter(zero_crossing_times[:500], np.zeros_like(zero_crossing_times)[:500], color='red', label='Zero Crossings')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Pole Tip Horizontal Offset")
    plt.legend()
    plt.title("Pole Tip Oscillation and Zero Crossings")
    plt.grid(True)
    plt.show()

    
    # 假設你已有 zero_crossing_times，如：
    # zero_crossing_times = np.array([...])
    
    # Step 1: 取偶數 index 的過零時間點
    even_crossing_times = zero_crossing_times[::2]
    # Step 2: 計算每個週期的時間長度
    periods = np.diff(even_crossing_times)  # 相鄰時間差，應該接近週期
    N = len(periods)
    dt = 1  # 每個樣本對應一個週期，不需要真實時間尺度
    
    # Step 3: FFT 分析
    yf = fft(periods)  # 去除 DC 分量 (均值)
    xf = fftfreq(N, dt)
    
    # 只取正頻率部分
    xf_pos = xf[:N//2]
    power_spectrum = np.abs(yf[:N//2])**2
    
    # Step 4: 繪圖
    plt.figure(figsize=(10, 5))
    plt.plot(xf_pos, power_spectrum)
    plt.xlabel("Frequency (1/crossings)")
    plt.ylabel("Power")
    plt.title("Fourier Spectrum of Zero-Crossing Periods")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    # 假設 you have `periods` = np.diff(even_crossing_times)
    periods = periods - np.mean(periods)  # 去掉均值
    N = len(periods)
    dt = 1  # index 間隔
    
    yf = fft(periods)
    xf = fftfreq(N, dt)
    xf = xf[:N//2]
    power = np.abs(yf[:N//2])**2
    Q = N
    # 避免 log(0)，加個 epsilon
    epsilon = 1e-12
    log_xf = np.log10(xf[2:Q] + epsilon)
    log_power = np.log10(power[2:Q] + epsilon)
    
    plt.figure(figsize=(8, 5))
    plt.plot(log_xf, log_power, label='Power Spectrum (log-log)')
    plt.xlabel('log10(Frequency)')
    plt.ylabel('log10(Power)')
    plt.title('Low-frequency behavior of Power Spectrum')
    plt.grid(True)
    plt.legend()
    plt.show()

# test
path = "PoleLength_0.5_PoleMass_0.05_Friction_0.npz"
fourier_analysis(path)