import taichi as ti
from src.clothes.Cloth import Cloth 

class GridCloth(Cloth):
    """
    init Grid Cloth
        N : tuple(N, M)
            NxM grid
        S : tuple(SN, SM)
            sizeX x sizeY in real coords
        KS : float
            stretching coeff
        KB : float
            Bending coeff
        KC : float
            collision coeff
        DAMPING : float
            damping
        center : ti.Vector([x, y, z])
            center of the cloth
        color : tuple(r, g, b)
            rgb color of the cloth
    """ 
    def __init__(self, N=(64, 64), S=(1, 1), KS=0.6, KB=0.4, KC=0.4, DAMPING=0.3, center=ti.Vector([0, 0, 0]), color=(1, 1, 1)):
        self.N, self.M = N
        self.SN, self.SM = S
        self.center = center

        self.KS = KS
        self.KB = KB
        self.KC = KC

        x, v = self.make_grid()

        x = self.gridTo1D(x)
        v = self.gridTo1D(v)

        i = self.make_indices()
        e = self.make_edges()
        t = self.make_tripairs()

        Cloth.__init__(self, self.N * self.M, x, v, i, e, t, color, KS, KC, KB, DAMPING)

    """
    flattens 2D array to 1D for super constructor
        a : 2D array
            either pos or velocity
    """
    def gridTo1D(self, a):
        res = ti.Vector.field(3, float, (self.N * self.M))
        for i, j in ti.ndrange(self.N, self.M):
            res[i * self.M + j] = a[i, j]
        return res

    """
    create grid
    """
    def make_grid(self):
        
        corner = self.center - ti.Vector([self.SN/2, self.SM/2, 0])

        x = ti.Vector.field(3, float, (self.N, self.M))
        v = ti.Vector.field(3, float, (self.N, self.M))
        
        for i, j in ti.ndrange(self.N, self.M):
            x[i, j] = corner + ti.Vector([i * self.SN / (self.N - 1), j * self.SM / (self.M - 1), 0])
            v[i, j] = ti.Vector([0, 0, 0]) 

        return x, v

    """
    create triangle list
    """
    def make_indices(self):
        num_triangles = (self.N - 1) * (self.M - 1) * 2
        indices = ti.field(int, num_triangles * 3)

        for i, j in ti.ndrange(self.N, self.M):
            if i < self.N - 1 and j < self.M - 1:
                square_id = (i * (self.M - 1)) + j
                # 1st triangle of the square
                indices[square_id * 6 + 0] = i * self.M + j
                indices[square_id * 6 + 1] = (i + 1) * self.M + j
                indices[square_id * 6 + 2] = i * self.M + (j + 1)
                # 2nd triangle of the square
                indices[square_id * 6 + 3] = (i + 1) * self.M + j + 1
                indices[square_id * 6 + 4] = i * self.M + (j + 1)
                indices[square_id * 6 + 5] = (i + 1) * self.M + j
        return indices

    def make_edges(self):
        edges_list = []
        
        for i, j in ti.ndrange(self.N, self.M):
            if i + 1 < self.N:
                edges_list.append(i * self.M + j)
                edges_list.append((i + 1) * self.M + j)

            if j + 1 < self.M:
                edges_list.append(i * self.M + j)
                edges_list.append(i * self.M + j + 1)

            if j + 1 < self.M and i + 1 < self.N:
                edges_list.append(i * self.M + j)
                edges_list.append((i + 1) * self.M + j + 1)

            if j + 1 < self.M and i + 1 < self.N:
                edges_list.append(i * self.M + j + 1)
                edges_list.append((i + 1) * self.M + j)

        l = len(edges_list)
        edges = ti.field(int, l)
        for i in range(l):
            edges[i] = edges_list[i]

        return edges

    def make_tripairs(self):
        tripair_list = []
        for i, j in ti.ndrange(self.N, self.M):
            if i + 1 < self.N and j+1 < self.M:
                tripair_list.append(i * self.M + j)
                tripair_list.append( (i+1) * self.M + j + 1)
                tripair_list.append( i * self.M + j + 1)
                tripair_list.append( (i+1) * self.M + j)

            if i + 1 < self.N and j+1 < self.M and i - 1 >= 0:
                tripair_list.append(i * self.M + j)
                tripair_list.append( (i) * self.M + j+1)
                
                tripair_list.append( (i+1) * self.M + j + 1)
                tripair_list.append( (i-1) * self.M + j)

            if i + 1 < self.N and j+1 < self.M and j - 1 >= 0:
                tripair_list.append(i * self.M + j)
                tripair_list.append( (i+1) * self.M + j)
                
                tripair_list.append( i * self.M + j - 1)
                tripair_list.append( (i+1) * self.M + j + 1)

        l = len(tripair_list)
        tripairs = ti.field(int, l)
        for i in range(l):
            tripairs[i] = tripair_list[i]

        return tripairs