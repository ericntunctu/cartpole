import numpy as np
length = float(input("Length: "))
mass = float(input("Mass: "))
friction_coef = float(input("Friction_Coef: "))

data = np.load(f"PoleLength_{length}_PoleMass_{mass}_Friction_{friction_coef}.npz")

for key in data.files:
    print(f"--- {key} ---")
    print(data[key])
    print()