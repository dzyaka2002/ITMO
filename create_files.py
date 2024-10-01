import random
import os

circle = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

square = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

triangle = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0]
]

files = [] + ['c:/python/1lab/train/circle.txt', 'c:/python/1lab/train/square.txt', 'c:/python/1lab/train/triangle.txt']
for i in files:
        if not os.path.exists(i):
            with open(i, 'w') as f:
                if 'circle' in i:
                    for row in circle:
                        f.write(' '.join(map(str, row)) + '\n')
                if 'square' in i:
                    for row in square:
                        f.write(' '.join(map(str, row)) + '\n')
                if 'triangle' in i:
                    for row in triangle:
                        f.write(' '.join(map(str, row)) + '\n')

def generate_sample(name, postfix, dir):
    f = open('c:/python/1lab/train/' + name + '.txt', 'r')
    res = f.readline()
    matrix = []
    for line in f:
        matrix.append(list(map(int, line.split())))
    f.close()
    for _ in range(5):
        x = random.randint(0, 5)
        y = random.randint(0, 5)
        matrix[x][y] = 1 - matrix[x][y]
    f = open(dir + '/' + name + '_' + str(postfix) + '.txt', 'w')
    f.write(res)
    for line in matrix:
        f.write(" ".join(list(map(str, line))))
        f.write('\n')
    f.close()

for i in range(1,40):
    generate_sample('circle', i, 'c:/python/1lab/train/')
    generate_sample('square', i, 'c:/python/1lab/train/')
    generate_sample('triangle', i, 'c:/python/1lab/train/')
    generate_sample('circle', i, 'c:/python/1lab/test')
    generate_sample('square', i, 'c:/python/1lab/test')
    generate_sample('triangle', i, 'c:/python/1lab/test')
