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
    def __init__(self, center, radius, color=(1, 1, 1)):
        super()
        self.center = center
        self.radius = radius
        self.color = color

        self.center_field = ti.Vector.field(3, float, (1, ))
        r = ti.Vector([0, 0, -self.radius])
        self.center_field[0] = self.center - r


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