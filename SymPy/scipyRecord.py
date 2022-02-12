# -*- coding: utf-8 -*-

import scipy.integrate as integrate
import scipy.special as special
result = integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)
result

#%% matplotlib绘图
import numpy as np
import matplotlib.pyplot as plt

Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)

fig0, ax0 = plt.subplots()
strm = ax0.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)

fig1, (ax1, ax2) = plt.subplots(ncols=2)  # 即有两张子图并列排放
ax1.streamplot(X, Y, U, V, density=[0.5, 1])

lw = 5*speed / speed.max()
ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)

plt.show()

#%% matplotlib草稿
# 绘制马鞍面 z=x^2/9 - y^2/4
from mpl_tookkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

X, Y = np.mgrid[-3:3:100j, -3:3:100j]
Z = X**2/9 - Y**2/4
fig = plt.figure()

ax.plot_surface(X, Y, Z)
# ax.streamplot(X, Y, Z, color=Z)
# plt.show()