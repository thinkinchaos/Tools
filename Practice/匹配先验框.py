import sys

tmp = sys.stdin.readline().strip().split(' ')
tmp = [int(i) for i in tmp]
w, h, s, t, k, p, q = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6]


def overlap(box1, box2):
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
        return False
    else:
        if (maxx - minx) * (maxy - miny) > 0:
            return True
        else:
            return False


priors = []
for i in range(0, p - w + 1, s):
    for j in range(0, q - h + 1, t):
        prior = [i, j, i + w, j + h, False]
        priors.append(prior)

cnt = 0
for _ in range(k):
    box = sys.stdin.readline().strip().split(' ')
    box = [int(i) for i in box]
    X, Y, W, H = box[0], box[1], box[2], box[3]
    box_ = [X, Y, X + W, Y + H]
    for prior in priors:
        if prior[4] is False:
            if overlap(prior[:4], box_):
                cnt += 1
            prior[4] = True
print(cnt)