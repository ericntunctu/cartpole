import numpy as np

def Sine_Terrain(x, amplitude = 1, frequency = 1, trannslation = 0):
    return amplitude * np.sin(frequency * (x - trannslation))

def Slope_Terrain(x, slope = 1, translation = 0):
    return slope * (x - translation)

class Terrain:
    def __init__(self):
        self.terrain = []
        self.Function = {
            'Sine': Sine_Terrain,
            'Slope': Slope_Terrain
        }

    def ask_user(self):
        print(f'Avaliable Function: {list(self.Function.keys())}')
        
        while True:
            func_name = input('Please input the function(input stop to stop): ').strip()
            if func_name == 'stop':
                break
            print(f"\nSetting  the parameter of {func_name}")

            if func_name == 'Sine':
                print('Sine Parameter Example: (amplitude, frequency, translation)')
                while True:
                    Parameter = input('Sine Parameter: ').strip()
                    if Parameter == 'skip':
                        break
                    A, k, tx = map(float, Parameter.strip('()').split(','))
                    times = int(input('Times: ').strip())
                    self.terrain.append({
                        'Terrain Name': func_name,
                        'Parameter': {'amplitude': A, 'frequency': k, 'trannslation': tx},
                        'Time': times
                    })
                    break
            else:
                print('Slope Parameter Example: (slope, translation)')
                while True:
                    Parameter = input('Slope Parameter: ').strip()
                    if Parameter == 'skip':
                        break
                    m, tx = map(float, Parameter.strip('()').split(','))
                    times = int(input('Times: ').strip())
                    self.terrain.append({
                        'Terrain Name': func_name,
                        'Parameter': {'slope': m, 'translation': tx},
                        'Time': times
                    })
                    break

    # 取得第 index 筆地形的可呼叫函式（h(x)）
    def get_fn(self, index=0):
        item = self.terrain[index]
        name = item['Terrain Name']
        params = item['Parameter']
        if name == 'Sine':
            def fn(x, _p=params):
                return Sine_Terrain(x, **_p)
        else:
            def fn(x, _p=params):
                return Slope_Terrain(x, **_p)
        return fn

    # 取得第 index 筆地形的標籤（給訓練程式檔名用）
    def get_tag(self, index=0):
        item = self.terrain[index]
        name = item['Terrain Name']
        p = item['Parameter']
        if name == 'Sine':
            return f"sin_A{p['amplitude']}_k{p['frequency']}_tx{p['trannslation']}"
        else:
            return f"slope_m{p['slope']}_tx{p['translation']}"
