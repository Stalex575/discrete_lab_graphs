"""
Lab 2 by Stadnik Oleksandr and Yaryna Zabchuk
"""
from collections import deque

def read_incidence_matrix(filename: str) -> list[list]:
    """
    Stanik Oleksandr
    :param str filename: path to file
    :returns list[list]: the incidence matrix of a given graph
    """
    with open(filename, 'r', encoding='utf-8') as file:
        file = file.readlines()[1:-1]
        edges = []
        vertices = set()
        for line in file:
            line = line.strip().replace("->", '').replace(';', '').split()
            start, end = int(line[0]), int(line[1])
            vertices.add(start)
            vertices.add(end)
            edges.append((start, end))

        vertices = sorted(vertices)
        matrix = [[0 for _ in range(len(edges))] for _ in range(len(vertices))]

        for edge_index, (start, end) in enumerate(edges):
            if start == end:
                matrix[start][edge_index] = 2
            else:
                matrix[start][edge_index] = 1
                matrix[end][edge_index] = -1
        return matrix


def read_adjacency_matrix(filename: str) -> list[list]:
    """
    Yaryna Zabchuk
    :param str filename: path to file
    :returns list[list]: the adjacency matrix of a given graph
    """
    with open(filename, 'r', encoding='utf-8') as file:
        file = file.readlines()[1:-1]
        pair_tops = []
        all_tops = set()

        for line in file:
            line = line.strip().replace("->", '').replace(';', '').split()
            top1, top2 = int(line[0]), int(line[1])
            pair_tops.append((top1, top2))
            all_tops.add(top1)
            all_tops.add(top2)

        x = max(all_tops) + 1
        res = [[0 for _ in range(x)] for _ in range(x)]

        for top1, top2 in pair_tops:
            res[top1][top2] = 1

        return res


def read_adjacency_dict(filename: str) -> dict[int, list[int]]:
    """
    Yanryna Zabchuk
    :param str filename: path to file
    :returns dict: the adjacency dict of a given graph
    """
    res = {}
    with open(filename, 'r', encoding='utf-8') as file:
        file = file.readlines()[1:-1]
        for line in file:
            line = line.strip().replace("->", '').replace(';', '').split()
            top1, top2 = int(line[0]), int(line[1])
            if top1 not in res:
                res[top1] = [top2]
            else:
                res[top1].append(top2)
    return res


def iterative_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    Yaryna Zabchuk
    :param list[list] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    num_vertises = len(graph)

    visited = [0] * num_vertises
    stack = [start]
    res = []

    while len(stack) != 0:
        v = stack[0]
        stack.remove(v)
        visited[v] = True
        res.append(v)
        for n in range(num_vertises):
            if n in graph[v] and n not in res and n not in stack:
                stack.append(n)

    return res


def iterative_adjacency_matrix_dfs(graph: list[list], start: int) ->list[int]:
    """
    Yanryna Zabchuk
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    num_vertises = len(graph)

    visited = [0] * num_vertises
    stack = [start]
    res = []

    while len(stack) != 0:
        v = stack[0]
        stack.remove(v)
        visited[v] = True
        res.append(v)
        for n in range(num_vertises):
            if graph[v][n] and n not in res and n not in stack:
                stack.append(n)

    return res


def recursive_adjacency_dict_dfs(graph: dict[int, list[int]], start: int, is_oriented: bool) -> list[int]:
    """
    Yaryna Zabchuk
    :param list[list] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0, 0)
    [0, 1, 2]
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0, 1)
    [0, 1, 2, 3]
    """
    num_vertises = len(graph)

    visited = [0] * num_vertises
    res = []

    def dfs(v, res):
        """
        Performs a depth-first search (DFS) starting from vertex v, 
        adding all visited vertices to the connected component.
        """
        visited[v] = True
        res.append(v)
        for neighbor in range(num_vertises):
            if neighbor in graph[v] and not visited[neighbor]:
                dfs(neighbor, res)

    dfs(start, res)

    return res


def recursive_adjacency_matrix_dfs(graph: list[list[int]], start: int) ->list[int]:
    """
    Yanryna Zabchuk
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    num_vertises = len(graph)

    visited = [0] * num_vertises
    res = []

    def dfs(v, res):
        """
        Performs a depth-first search (DFS) starting from vertex v, 
        adding all visited vertices to the connected component.
        """
        visited[v] = True
        res.append(v)
        for n in range(num_vertises):
            if graph[v][n] and not visited[n]:
                dfs(n, res)

    dfs(start, res)

    return res


def iterative_adjacency_dict_bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    Stadnik Oleksandr
    :param list[list] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    visited = set()
    queue = deque([start])
    visited.add(start)
    bfs_order = []

    while queue:
        vertex = queue.popleft()
        bfs_order.append(vertex)
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                queue.append(neighbour)
                visited.add(neighbour)
    return bfs_order


def iterative_adjacency_matrix_bfs(graph: list[list[int]], start: int) ->list[int]:
    """
    Stanik Oleksandr
    :param dict graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    visited = set()
    queue = deque([start])
    visited.add(start)
    bfs_order = []

    while queue:
        vertex = queue.popleft()
        bfs_order.append(vertex)
        for neighbour in range(len(graph)):
            if graph[vertex][neighbour] and neighbour not in visited:
                queue.append(neighbour)
                visited.add(neighbour)
    return bfs_order


def adjacency_matrix_radius(graph: list[list]) -> int:
    """
    :param list[list] graph: the adjacency matrix of a given graph
    :returns int: the radius of the graph
    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    1
    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    2
    """
    pass


def adjacency_dict_radius(graph: dict[int: list[int]]) -> int:
    """
    :param dict graph: the adjacency list of a given graph
    :returns int: the radius of the graph
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1]})
    1
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [1]})
    2
    """
    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
