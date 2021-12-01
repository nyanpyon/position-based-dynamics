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
    def __init__(self, center, radius, color=(1, 1, 1), drest = 0.005):
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
        scene.particles(self.center_field, radius=self.radius*0.95, color=self.color)

    """
    @OVERRIDE
    check if p collides with the object
        p : ti.Vector([x, y, z])
            the point which collides
    """
    @ti.func
    def collides(self, p : ti.template(), old_p : ti.template()):
        d = p - old_p

        oc = old_p - self.center

        a = d.dot(d)
        b = 2 * oc.dot(d)
        c = oc.dot(oc) - (self.radius) * (self.radius) 

        t = -1.
        det = b * b - 4 * a * c 

        if det > 0:
            t1 = (-b + ti.sqrt(det)) / (2 * a)
            t2 = (-b - ti.sqrt(det)) / (2 * a)



            if t1 >= 0 and t2 >= 0:
                if t1 > t2:
                    t = t2
                else:
                    t = t1
        
        return t

    """
    @OVERRIDE
    solves the collision constraint for a point p
        p : ti.Vector([x, y, z])
            the point which collides
    """
    @ti.func
    def solve_collision_constraint(self, p : ti.template(), old_p : ti.template(), t : ti.f32):
        d =  p - old_p
        collision_point = old_p + t * d

        n = collision_point - self.center
        n = n / n.norm()
        
        C = (p - collision_point).dot(n) - self.drest
        lagrange = C
        return - lagrange * n