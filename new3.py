N = int(input())

MAP = []

C_POS = []
SCORE = [[0 for _ in range(N)] for _ in range(N)]

for ldx,line in enumerate(range(N)):
    _input = input().split()
    for jdx,char in enumerate(_input):
        if char == "C":
            C_POS.append((jdx,ldx))
        elif char == "A":
            SCORE[ldx][jdx] = 20000
    MAP.append(_input)



def out_of_range(pos):
    x,y = pos
    if x<0 or y<0:return True
    elif x>=N or y>=N:return True
    return False

def is_char(pos,char="A"):
    x,y = pos
    return MAP[y][x] == char

vec_4 = [
    [1,0],
    [-1,0],
    [0,1],
    [0,-1]
]

for cpoint in C_POS:
    old_point = set()
    queue = [(cpoint,0),]
    while queue:
        node,step = queue.pop(0)
        nx,ny = node
        for vx,vy in vec_4:
            _x,_y = nx+vx,ny+vy
            if out_of_range((_x,_y)) or not is_char((_x,_y)):
                continue
            elif (_x,_y) in old_point:
                continue
            old_point.add((_x,_y))
            queue.append(((_x,_y),step+1))
            SCORE[_y][_x] = min(SCORE[_y][_x],step+1)
total = 0
for i in sum(SCORE,[]):
    if i != 20000:
        total += i

print(total)
            
n2
root = Node(xxx)

current = root

for _ in range(100):
    current.next = Node(xxx)
    current = current.next