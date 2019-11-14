import random

f = lambda x,y,z : 1 if abs(x) + abs(y) + abs(z) < 1 else 0;


for i in range(10000):
    x = random.random();
    y = random.random();
    z = random.random();
    o = f(x,y,z);
    print("%.6f,%.6f,%.6f,%d," % (x,y,z,o));

