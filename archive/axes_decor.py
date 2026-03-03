#%%
import matplotlib.pyplot as plt
from lablib102 import extend_Axes_methods
plt.Axes = extend_Axes_methods(plt.Axes) # decorate Axes, 添加右轴上色方法

fig, ax = plt.subplots()
axx = ax.twinx()
axx.color_right_yax("r") # 右轴上色