import taichi as ti
from abc import ABCMeta, abstractmethod

@ti.data_oriented
class Object(object, metaclass=ABCMeta):
    def __init__(self):
        self.data
        self.result

    """
    @ABSTRACT METHOD
    draws the object
        scene: ti.scene
            the scene object
    """
    @abstractmethod
    def draw(self, scene):
        pass

    """
    @ABSTRACT METHOD
    check if p collides with the object
        p : ti.Vector([x, y, z])
            the point which collides
    """
    @abstractmethod
    def collides(self, p):
        pass

    """
    @ABSTRACT METHOD
    solves the collision constraint for a point p
        p : ti.Vector([x, y, z])
            the point which collides
    """
    @abstractmethod
    def solve_collision_constraint(self, p):
        pass

    def set_data(self, data, result):
        super.data = data
        super.result = result

    @ti.kernel
    def solve_collision_constraint_for_all(self):
        for i in range(super.data.shape[0]):
            if self.collides(super.data[i]):
                super.result[i] += self.solve_collision_constraint(super.data[i])
