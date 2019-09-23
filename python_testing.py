import numpy as np

a = np.array([[1, 2], [1, 2]])
b = []
for _ in range(3):
    b.append(a)

b = np.stack(b, axis=2) 
print(b.shape)
