import sys
from heapq import heappush, heappop
from collections import defaultdict

def solve(edgeList, trainIndices, n):
    graph = defaultdict(list)

    for i in range(len(edgeList)):
        l = edgeList[i][0]
        r = edgeList[i][1]
        c = edgeList[i][2]
        graph[l].append((r, c, i))
        graph[r].append((l, c, i))

    q = [(0, 0)]
    dists = [float('inf')] * n
    dists[0] = 0
    prev = [-1] * n
    prev_edge = [-1] * n

    while q:
        dist, node = heappop(q)

        if dist > dists[node]:
            continue

        for neighbor, cost, eidx in graph[node]:
            if dists[neighbor] > dist + cost:
                dists[neighbor] = dist + cost
                prev[neighbor] = node
                prev_edge[neighbor] = eidx
                heappush(q, (dist + cost, neighbor))
            elif dists[neighbor] == dist + cost:
                if prev_edge[neighbor] in trainIndices and eidx not in trainIndices:
                    prev[neighbor] = node
                    prev_edge[neighbor] = eidx

    used_trains = set()
    for i in range(1, n):
        node = i
        while node != 0:
            if prev[node] == -1:
                break
            if prev_edge[node] in trainIndices:
                used_trains.add(prev_edge[node])
            node = prev[node]

    return len(trainIndices) - len(used_trains)

def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1

    edgeList = []
    for _ in range(m):
        u = int(data[idx]) - 1; idx += 1
        v = int(data[idx]) - 1; idx += 1
        x = int(data[idx]); idx += 1
        edgeList.append((u, v, x))

    trainIndices = set()
    for _ in range(k):
        s = int(data[idx]) - 1; idx += 1
        y = int(data[idx]); idx += 1
        trainIndices.add(len(edgeList))
        edgeList.append((0, s, y))

    print(solve(edgeList, trainIndices, n))

main()