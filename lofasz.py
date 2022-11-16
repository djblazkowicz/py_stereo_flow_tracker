import numpy as np

filepath = './calib.txt'
with open(filepath, 'r') as f:
    params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
    P_l = np.reshape(params, (3, 4))
    K_l = P_l[0:3, 0:3]
    params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
    P_r = np.reshape(params, (3, 4))
    K_r = P_r[0:3, 0:3]
print('K_l')
print(K_l)
print('P_l')
print(P_l)
print('K_r')
print(K_r)
print('P_r')
print(P_r)
