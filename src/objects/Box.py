import taichi as ti
from src.objects.Object import Object 


"""
contains a list with the corresponding vertices for every face of the box
"""
CUBE_FACES = [
    [0, 1, 2, 3],
    [4, 7, 6, 5],
    [0, 4, 5, 1],
    [3, 2, 6, 7],
    [1, 5, 6, 2],
    [0, 3, 7, 4]
]

def make_rotation_matrix(rotation):
    a = rotation[0]
    b = rotation[1]
    c = rotation[2]
    R = ti.Matrix([
            [ti.cos(a) * ti.cos(b), ti.cos(a) * ti.sin(b) * ti.sin(c) - ti.sin(a) * ti.cos(c), ti.cos(a) * ti.sin(b) * ti.cos(c) + ti.sin(a) * ti.sin(c)],
            [ti.sin(a) * ti.cos(b), ti.sin(a) * ti.sin(b) * ti.sin(c) + ti.cos(a) * ti.cos(c), ti.sin(a) * ti.sin(b) * ti.cos(c) - ti.cos(a) * ti.sin(c)],
            [- ti.sin(b), ti.cos(b) * ti.sin(c), ti.cos(b) * ti.cos(c)]
        ])
    return R
"""
divides a face into 2 triangles
"""
def make_tri_from_quad(list):
    return [list[0], list[2], list[1], list[0], list[2], list[3]]
    
"""
returns the list with all the triangles of a box
"""
def get_cube_indices():
    res = [make_tri_from_quad(x) for x in CUBE_FACES]
    return [item for sublist in res for item in sublist]


class Box(Object):

    """
    inits a Box
        center : ti.Vector([x, y, z]) 
            center of the box
        size : ti.Vector(widht, depth, height)
            contains x size, y size and z size of the box
        color : tuple(r, g, b) , optional
            rgb color of the cube
    """
    def __init__(self, center, size, color=(1, 1, 1), drest=0.01, rotation=[0, 0, 0]):
        self.center = center
        self.size = size
        self.color = color
        self.drest = drest
        
        
        
        self.vertices, self.indices = self.make_box()
        self.rotation = rotation
        
        self.R = make_rotation_matrix(self.rotation)
        print(self.R)
        self.rotate_box()

    """
    returns the vertices list and the indeces of the box
    used to draw the triangle mesh
    """
    def make_box(self):
        vertices = ti.Vector.field(3, float, 8)

        w = ti.Vector([self.size.x, 0, 0])
        d = ti.Vector([0, self.size.y, 0])
        h = ti.Vector([0, 0, self.size.z])

        vertices[0] = self.center - (self.size/2)
        vertices[1] = vertices[0] + w
        vertices[2] = vertices[1] + h
        vertices[3] = vertices[2] - w
        vertices[4] = vertices[0] + d
        vertices[5] = vertices[4] + w
        vertices[6] = vertices[5] + h
        vertices[7] = vertices[6] - w

        indices = ti.field(int, 3 * 12)
        for i, e in enumerate(get_cube_indices()):
            indices[i] = e
        return vertices, indices

    @ti.kernel
    def rotate_box(self):
        for i in range(self.vertices.shape[0]):
            self.vertices[i] = (self.R @ (self.vertices[i] - self.center)) + self.center
    """
    @OVERRIDE
    draws the object
        scene: ti.scene
            the scene object
    """
    def draw(self, scene):
        scene.mesh(self.vertices,
               indices=self.indices,
               color=self.color,
               two_sided=True)


    """
    @OVERRIDE
    solves the collision constraint for a point p
        p : ti.Vector([x, y, z])
            the point which collides
    """
    @ti.func
    def solve_collision_constraint(self, p : ti.template(), x : ti.template()):
        cp = p - self.center
        w = self.R @ ti.Vector([self.size.x/2, 0, 0]) 
        d = self.R @ ti.Vector([0, self.size.y/2, 0])
        h = self.R @ ti.Vector([0, 0, self.size.z/2])
        what = w/w.norm() 
        dhat = d/d.norm()
        hhat = h/h.norm()

        dw = cp.dot(what)
        dd = cp.dot(dhat)
        dh = cp.dot(hhat)

        absdw = w.norm() + self.drest - abs(dw)
        absdd = d.norm() + self.drest - abs(dd)
        absdh = h.norm() + self.drest - abs(dh)

        res = x - x
        if absdw > 0 and absdd > 0 and absdh > 0:
            argmin = 1
            vmin = absdw
            if vmin > absdd:
                argmin = 2
                vmin = absdd
            if vmin > absdh:
                argmin = 3
                vmin = absdh
            
            if argmin == 1:
                res = what * (absdw)
                if dw < 0:
                    res = - res
            elif argmin == 2:
                res = dhat * (absdd)
                if dd < 0:
                    res = - res
            elif argmin == 3:
                res = hhat * (absdh)
                if dh < 0:
                    res = - res
        return res
        