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
    def __init__(self, center, size, color=(1, 1, 1)):
        super()
        self.center = center
        self.size = size
        self.color = color

        self.vertices, self.indices = self.make_box()

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
    check if p collides with the object
        p : ti.Vector([x, y, z])
            the point which collides
    """
    def collides(self, p):
        pass

    """
    @OVERRIDE
    solves the collision constraint for a point p
        p : ti.Vector([x, y, z])
            the point which collides
    """
    def solve_collision_constraint(self, p):
        pass