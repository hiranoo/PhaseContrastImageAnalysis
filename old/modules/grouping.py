import numpy as np
from numba import jit


@jit(nopython=True)
def validate(x, begin_x, end_x):
    return begin_x <= x and x < end_x

@jit(nopython=True)
def dfs(x, y, src, ksize, groups, stopped, group_id, depth):
    # 大きすぎる領域は探索を打ち切る
    if depth >= 1000:
        stopped[group_id] = 1
        return
    for dy in range(-ksize//2, ksize//2 + 1):
        if (not validate(y + dy, 0, src.shape[0])):
            continue
        for dx in range(-ksize//2, ksize//2 + 1):
            if (not validate(x + dx, 0, src.shape[1])):
                continue
            if (dx == 0 and dy == 0):
                continue
            ny, nx = y + dy, x + dx
            if (src[ny, nx] > 0 and groups[ny, nx] == 0):
                groups[ny, nx] = group_id
                dfs(x + dx, y + dy, src, ksize, groups, stopped, group_id, depth + 1)

def divideIntoGroups(image, ksize):
    grouped = np.zeros(image.shape, dtype=int)
    stopped = np.zeros(grouped.shape[0] * grouped.shape[1] + 10)
    group_id = 1
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            if (image[y, x] > 0 and grouped[y, x] == 0):
                grouped[y, x] = group_id
                dfs(x, y, image, ksize, grouped, stopped, group_id, 0)
                group_id += 1
    
    group_sizes = np.zeros(group_id)
    for y in range(0, grouped.shape[0]):
        for x in range(0, grouped.shape[1]):
            group_id = int(grouped[y, x])
            # 探索を打ち切ったグループはサイズ０のまま、グループIDを-1にする
            if stopped[group_id] > 0:
                grouped[y, x] = -1
                continue
            group_sizes[group_id] += 1
    return (group_sizes, grouped)

@jit(nopython=True)
def _count_label_sizes(nlabels, labels):
    res = np.zeros(nlabels)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label_id = labels[y, x]
            if label_id > 0:
                res[label_id] += 1
    return res

def count_label_sizes(nlabels, labels):
    return _count_label_sizes(nlabels, labels)