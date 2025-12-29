from typing import cast


def read_graph(filename) -> tuple[range, list[set[int]]]:
    if len(filename)>6 and filename[-6:] == ".col.b":
        return read_graph_binary(filename)
    else:
        return read_graph_ascii(filename)

def bytes_to_bits(bytes: bytes) -> list[int]:
    bits = []
    i = 0
    j = 0

    for byte in bytes:
        for _ in range(8):
            bits.append((byte & 0b10000000) >> 7)
            byte <<= 1
            j += 1

            if j > i:
                i += 1
                j = 0
                break

    return bits

def read_edges(line: bytes, adj: list[set[int]]) -> None:
    adj_mat = bytes_to_bits(line)
    i = 0
    j = 0

    for entry in adj_mat:        
        if entry == 1:
            adj[i].add(j)
            adj[j].add(i)

        j += 1

        if i < j:
            i += 1
            j = 0


def read_graph_binary(filename) -> tuple[range, list[set[int]]]:
    """Read a graph from a file in the Binary format specified by David Johnson
    for the DIMACS clique challenge.
    Instances are available at
    ftp://dimacs.rutgers.edu/pub/challenge/graph/benchmarks/clique
    """
    try:
        if len(filename)>3 and filename[-3:] == ".gz":  # file compressed with gzip
            import gzip
            f = gzip.open(filename, "rb")
        else:   # usual, uncompressed file
            f = open(filename, "rb")
    except IOError:
        print("could not open file", filename)
        exit(-1)

    preamble_size = int(f.readline())
    preamble = f.read(preamble_size)

    if type(preamble) is bytes:
        preamble = preamble.decode()

    preamble = cast(str, preamble)

    last_line = preamble.split("\n")[-2]
    p, name, n, nedges = last_line.split()
    n, nedges = int(n), int(nedges)
    nodes = range(n)
    adj: list[set[int]] = [set([]) for i in nodes]
    binary_line = f.read()

    if type(binary_line) is str:
        binary_line = binary_line.encode()
    binary_line = cast(bytes, binary_line)

    read_edges(binary_line, adj)

    f.close()
    return nodes, adj

def read_graph_ascii(filename) -> tuple[range, list[set[int]]]:
    """Read a graph from a file in the ASCII format specified by David Johnson
    for the DIMACS clique challenge.
    Instances are available at
    ftp://dimacs.rutgers.edu/pub/challenge/graph/benchmarks/clique
    """
    try:
        if len(filename)>3 and filename[-3:] == ".gz":  # file compressed with gzip
            import gzip
            f = gzip.open(filename, "rb")
        else:   # usual, uncompressed file
            f = open(filename)
    except IOError:
        print("could not open file", filename)
        exit(-1)

    nodes: range = range(0) # For typing
    adj: list[set[int]] = [] # For typing

    for line in f:
        if line[0] == 'e':
            e, i, j = line.split()
            i,j = int(i)-1, int(j)-1 # -1 for having nodes index starting on 0
            if i == j: # The node has a self-loop
                continue 
            adj[i].add(j)
            adj[j].add(i)
        elif line[0] == 'c':
            continue
        elif line[0] == 'p':
            p, name, n, nedges = line.split()
            n, nedges = int(n), int(nedges)
            nodes = range(n)
            adj = [set([]) for i in nodes]
    f.close()
    return nodes, adj