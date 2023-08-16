# %%
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def func(x, a, b):
    return a * np.exp(-x * b)


x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([0.8, 0.15, 0.02, 0.008])
y_2 = np.array([0.4, 0.4, 0.1, 0.05, 0.01, 0.01])
x_extended = np.array(list(x for x in range(20)))
popt, pcov = curve_fit(func, x, y_2)
""""""
plot_1 = plt.bar(
    x_extended,
    func(x_extended, *popt),
    label="fit: a=%5.3f, b=%5.3f" % tuple(popt),
)
plt.show()
plt.clf()
plot_2 = plt.bar(
    x_extended,
    np.array([0.4, 0.4, 0.1, 0.05, 0.01, 0.01] + (len(x_extended) - len(y_2)) * [0.01]),
    label="fit: a=%5.3f, b=%5.3f" % tuple(popt),
)
plt.show()
print(func(x_extended, *popt))
