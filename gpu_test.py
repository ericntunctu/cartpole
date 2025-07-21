
import torch
import time

# 檢查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Device name:", torch.cuda.get_device_name(0))

# 建立一個大張 tensor 並搬到 GPU
a = torch.randn(5000, 5000, device=device)
b = torch.randn(5000, 5000, device=device)

# 用 GPU 計時矩陣乘法
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()  # 等待 GPU 完成運算
end = time.time()

print("Matrix multiplication result:", c)
print("Time taken on GPU:", round(end - start, 4), "seconds")
