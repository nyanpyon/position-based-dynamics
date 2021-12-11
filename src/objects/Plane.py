import taichi as ti
from src.objects.Object import Object 

class Plane(Object):

    """
    init Sphere
        center : ti.Vector([x, y, z])
            center of the sphere
        radius : int
            radius of the sphere
        color : tuple(r, g, b), optional
            rgb color of the sphere range[0-1]
    """
    def __init__(self, center, normal=ti.Vector([0, 0, 1]), S =1000, color=(1, 1, 1), drest = 0.001):
        self.center = center
        self.color = color
        self.drest = drest
        self.S = S
        self.normal = normal/normal.norm()

        self.center_field = ti.Vector.field(3, float, (1, ))
        
        self.center_field[0] = self.center
        self.vertices = ti.Vector.field(3, float, 5)


        s1 = self.perpendicular(self.normal)
        s1 = s1/s1.norm() * self.S
        s2 = s1.cross(self.normal)
        s2 = s2/s2.norm() * self.S

        self.vertices[0] = self.center
        self.vertices[1] = self.center + s1 + s2
        self.vertices[2] = self.center - s1 + s2
        self.vertices[3] = self.center + s1 - s2
        self.vertices[4] = self.center - s1 - s2

        self.indices= ti.field(int, 3 * 4)
        indices = [0, 1, 2, 0, 2, 4, 0, 3, 1, 0, 3, 4]
        for i in range(12):
            self.indices[i] = indices[i]

    def perpendicular(self, a):
        if not (a[0] == 0 and a[1] == 0):
            return ti.Vector([-a[1], -a[0], 0])
        return ti.Vector([a[2], 0, a[0]])



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
    @ti.func
    def collides(self, p : ti.template(), old_p : ti.template()):
        b = -(old_p - self.center)
        d = p - old_p
        t = (b.dot(self.normal))/(d.dot(self.normal))
        return t

    """
    @OVERRIDE
    solves the collision constraint for a point p
        p : ti.Vector([x, y, z])
            the point which collides
    """
    @ti.func
    def solve_collision_constraint(self, p : ti.template(), old_p : ti.template(), t : ti.f32):
        d = p - old_p
        cp = old_p + t * d
        return -(1.1-t) * d

    @ti.func
    def push_outside(self, p : ti.template(), old_p : ti.template(), t : ti.f32):
        d = p - old_p
        return -t * d + self.drest * self.normal