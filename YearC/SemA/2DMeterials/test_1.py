import numpy as np

a = np.linspace(0, 10, 10)
b = np.linspace(0, 10*a[-1], 10*a.size)
print(a)
print(a.size)
print(b)
print(b.size)

c = np.concatenate(
    [
        a+a[-1]*i for i in range(11)
    ]
)

print(c)
print(c.size)