M,N,K = map(int,input().split())

MAP = [[0 for _ in range(N)] for _ in range(M)]

for line in range(K):
    x,y = map(int,input().split())
    MAP[x][y] = 1

sx,sy = map(int,input().split())
ex,ey = map(int,input().split())

Can_get = {}
old_point = set((sx,sy))

vec_8 = [
    [1,0],
    [-1,0],
    [0,1],
    [0,-1],
    [1,1],
    [1,-1],
    [-1,1],
    [-1,-1]
]

def out_of_range(pos):
    x,y = pos
    if x<0 or y<0:return True
    elif x>=M or y>=N:return True
    return False

def is_inside(pos):
    x,y = pos
    return MAP[x][y]

def get_can_get(pos,old_point):
    _pos_list = []
    x,y = pos
    for vx,vy in vec_8:
        _x = x+vx
        _y = y+vy
        step = False
        while not out_of_range((_x,_y)) and is_inside((_x,_y)):
            step = True
            _x = _x+vx
            _y = _y+vy
        if not out_of_range((_x,_y)) and step:
            if (_x,_y) not in old_point:
                _pos_list.append((_x,_y))
                old_point.add((_x,_y))
    return _pos_list,old_point

def find_route(_sx,_sy,old_point):
    if (_sx,_sy) == (ex,ey):
        return True,[(ex,ey)]
    if (_sx,_sy) not in Can_get:
        Can_get[(_sx,_sy)] = get_can_get((_sx,_sy))
    for _x,_y in Can_get[(_sx,_sy)]:
        ok,route = find_route(_x,_y)
        if ok:
            route.insert(0,(_sx,_sy))
            return True,route
    return False,None#不打包->..没有可执行文件，运行代码 设计模式 views多利
from typing import Optional
# 创建 挂载 销毁
ok,route = find_route(sx,sy,old_point)
if not ok:
    print(0)
else:
    print(len(route)-1)
    for x,y in route[1:]:
        print(x,y)