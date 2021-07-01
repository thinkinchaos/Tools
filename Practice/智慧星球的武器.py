import sys

K = int(sys.stdin.readline().strip())


def get_d(z, d, p):
    assert z in [1, 2, 3]
    if z == 1:
        t = d + p
    elif z == 2:
        t = d * p
    else:
        t = d // p
    return max(1, t)


d = 1
for _ in range(K):
    t = sys.stdin.readline().strip().split(' ')
    z, p = int(t[0]), int(t[1])
    d = get_d(z, d, p)

print(d)
