# animate_cartpole_from_npz.py
# 讀取訓練產生的 .npz 並播放動畫（支援地形 Sine / Slope；缺資料時視為平地）
# 需要: matplotlib, numpy, scipy(非必要), ffmpeg(若要輸出 mp4)

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ========= 使用者設定 =========
INPUT_NPZ  = r"C:\Users\kyleh\OneDrive\文件\HRV & CartPole\Cartpole_Terrain_Code\Cartpole_Terrain\Code\cartpole_data\Terrain_sin_A0.01_k0.01_tx0.0_PoleLength_1.0_PoleMass_1.0_Friction_0.npz"
OUTPUT_DIR = r"C:\Users\kyleh\OneDrive\文件\HRV & CartPole\Cartpole_Terrain_Code\Sine_Terrain_Data"
SAVE_VIDEO = False  # 想要輸出 mp4 就改成 True（需要電腦有 ffmpeg）
VIDEO_FPS  = 60     # 影片 fps（儲存時用）

os.makedirs(OUTPUT_DIR, exist_ok=True)
base = os.path.splitext(os.path.basename(INPUT_NPZ))[0]

# ========= 載入資料 =========
data = np.load(INPUT_NPZ, allow_pickle=True)

x_arr      = data["cart_positions"]
theta_arr  = data["pole_angles"]            # 與世界垂直的夾角（0=直立）
t_arr      = data["timestamps"]
phis       = data["phis"] if "phis" in data.files else np.zeros_like(x_arr)

terrain_name  = data["terrain_name"][0]      if "terrain_name" in data.files else ""
terrain_param = data["terrain_parameter"][0] if "terrain_parameter" in data.files else {}

# 訓練時間（如果存在就帶到畫面資訊）
train_secs    = float(data["training_seconds"]) if "training_seconds" in data.files else None
train_device  = str(data["training_device"][0]) if "training_device" in data.files else None
train_steps   = int(data["training_total_steps"]) if "training_total_steps" in data.files else None

# 解析桿長（完整長度），Env 內使用的是「半長」
m = re.search(r"PoleLength_([0-9.]+)", base)
full_length = float(m.group(1)) if m else 1.0
L = full_length / 2.0  # 以 env 規格，樞紐到桿端距離

# 嘗試從時間軸估 tau（也可手動指定）
if len(t_arr) > 1:
    tau = float(np.median(np.diff(t_arr)))
else:
    tau = 0.02

# ========= 還原地形函數 h(x) =========
def sine_terrain(x, amplitude=1.0, frequency=1.0, trannslation=0.0):
    return amplitude * np.sin(frequency * (x - trannslation))

def slope_terrain(x, slope=1.0, translation=0.0):
    return slope * (x - translation)

def get_terrain_fn(name, params):
    if str(name).lower().startswith("sine"):
        A  = float(params.get("amplitude", 0.0))
        k  = float(params.get("frequency", 0.0))
        tx = float(params.get("trannslation", 0.0))
        return lambda xx: sine_terrain(xx, A, k, tx)
    elif str(name).lower().startswith("slope"):
        m  = float(params.get("slope", 0.0))
        tx = float(params.get("translation", 0.0))
        return lambda xx: slope_terrain(xx, m, tx)
    else:
        return lambda xx: 0.0 * np.asarray(xx)

h = get_terrain_fn(terrain_name, terrain_param)

# ========= 視覺化元素尺度 =========
# 取 x 範圍，左右各留邊距
xmin = float(np.min(x_arr)) - 1.5
xmax = float(np.max(x_arr)) + 1.5

# 掃軌地形線用的 x 軸
xline = np.linspace(xmin, xmax, 1000)
yline = h(xline)

# 手推車/桿子的尺寸（視覺化用，不影響物理）
cart_w = 0.35
cart_h = 0.22
pivot_offset_y = cart_h / 2.0  # 樞紐在車頂中央

# ========= Matplotlib Figure =========
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(xmin, xmax)

# y 範圍：地形的極值 + 桿子的最大高度
y_terrain_min = float(np.min(yline))
y_terrain_max = float(np.max(yline))
ymax_est = y_terrain_max + pivot_offset_y + full_length * 1.2
ymin_est = y_terrain_min - 0.8
ax.set_ylim(ymin_est, ymax_est)

ax.set_xlabel("x")
ax.set_ylabel("y")
ttl = f"CartPole on Terrain | {terrain_name} {terrain_param}" if len(str(terrain_name))>0 else "CartPole on Terrain"
title_text = ax.set_title(ttl)

# 畫地形
terrain_line, = ax.plot(xline, yline, lw=1.5, alpha=0.8, label="terrain")

# 畫手推車（用一個矩形 + 中心點）
cart_patch = plt.Rectangle((x_arr[0] - cart_w/2, h(x_arr[0])), cart_w, cart_h,
                           fill=True, alpha=0.9, edgecolor="k")
ax.add_patch(cart_patch)

cart_center, = ax.plot([x_arr[0]], [h(x_arr[0])+cart_h/2], 'o', ms=4, label="cart center")

# 畫桿：用 Line2D 從樞紐到桿端
pole_line, = ax.plot([], [], lw=2.0, label="pole")

# 訊息文字（時間、步數、訓練資訊等）
info_text = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                    fontsize=10, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

ax.legend(loc="upper right")

# ========= 動畫更新函數 =========
def init():
    cart_patch.set_xy((x_arr[0] - cart_w/2, h(x_arr[0])))
    cart_center.set_data([x_arr[0]], [h(x_arr[0])+cart_h/2])
    x0 = x_arr[0]
    y0 = h(x0) + pivot_offset_y
    x_tip = x0 + L * np.sin(theta_arr[0])
    y_tip = y0 + L * np.cos(theta_arr[0])
    pole_line.set_data([x0, x_tip], [y0, y_tip])

    t0 = t_arr[0] if len(t_arr) > 0 else 0.0
    info_lines = [f"t = {t0:.2f} s | step = 0  ",
                  f"L (full) = {full_length:.3f} | tau ≈ {tau:.3f} s"]
    if train_secs is not None:
        info_lines.append(f"train = {train_secs:.1f} s | steps = {train_steps}")
    if train_device is not None:
        info_lines.append(f"device = {train_device}")
    info_text.set_text("\n".join(info_lines))
    return terrain_line, cart_patch, cart_center, pole_line, info_text

def update(frame):
    # 位置 / 地形
    x = float(x_arr[frame])
    y = float(h(x))
    # cart
    cart_patch.set_xy((x - cart_w/2, y))
    cart_center.set_data([x], [y + cart_h/2])

    # pole: 以世界垂直為 0 度，樞紐在車頂中央
    theta = float(theta_arr[frame])
    x0 = x
    y0 = y + pivot_offset_y
    x_tip = x0 + L * np.sin(theta)
    y_tip = y0 + L * np.cos(theta)
    pole_line.set_data([x0, x_tip], [y0, y_tip])

    # 畫面資訊
    t = float(t_arr[frame]) if frame < len(t_arr) else frame * tau
    info_lines = [f"t = {t:6.2f} s | step = {frame:7d}",
                  f"L (full) = {full_length:.3f} | tau ≈ {tau:.3f} s"]
    if train_secs is not None:
        info_lines.append(f"train = {train_secs:.1f} s | steps = {train_steps}")
    if train_device is not None:
        info_lines.append(f"device = {train_device}")
    info_text.set_text("\n".join(info_lines))

    return terrain_line, cart_patch, cart_center, pole_line, info_text

# ========= 建立動畫 =========
frames = len(x_arr)
# 避免資料過長播放太慢，可以下采樣（例如每 2~5 幀取 1）
# ds = 1
# x_arr = x_arr[::ds]; theta_arr = theta_arr[::ds]; t_arr = t_arr[::ds]; phis = phis[::ds]; frames = len(x_arr)

ani = animation.FuncAnimation(
    fig, update, init_func=init,
    frames=frames, interval=max(1, int(tau * 1000)), blit=True
)

# ========= 顯示或輸出 =========
if SAVE_VIDEO:
    mp4_path = os.path.join(OUTPUT_DIR, f"{base}_anim.mp4")
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=VIDEO_FPS, metadata=dict(artist='CartPole'), bitrate=1800)
        ani.save(mp4_path, writer=writer)
        print("Saved video:", mp4_path)
    except Exception as e:
        print("FFMPEG not available or save failed:", e)
        print("Falling back to live display...")
        plt.show()
else:
    plt.show()
