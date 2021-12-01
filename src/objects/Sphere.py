import taichi as ti
from src.objects.Object import Object 

class Sphere(Object):

    """
    init Sphere
        center : ti.Vector([x, y, z])
            center of the sphere
        radius : int
            radius of the sphere
        color : tuple(r, g, b), optional
            rgb color of the sphere range[0-1]
    """
    def __init__(self, center, radius, color=(1, 1, 1), drest = 0.01):
        self.center = center
        self.radius = radius
        self.color = color
        self.drest = drest

        self.center_field = ti.Vector.field(3, float, (1, ))
        
        self.center_field[0] = self.center



    """
    @OVERRIDE
    draws the object
        scene: ti.scene
            the scene object
    """
    def draw(self, scene):
        scene.particles(self.center_field, radius=self.radius, color=self.color)

    """
    @OVERRIDE
    check if p collides with the object
        p : ti.Vector([x, y, z])
            the point which collides
    """
    @ti.func
    def collides(self, p : ti.template()):
        cp = p - self.center
        
        return cp.norm() < self.radius + self.drest

    """
    @OVERRIDE
    solves the collision constraint for a point p
        p : ti.Vector([x, y, z])
            the point which collides
    """
    @ti.func
    def solve_collision_constraint(self, p : ti.template()):
        cp = p - self.center
        n = cp / cp.norm()
        return (cp.norm() - (self.radius+self.drest)) * n