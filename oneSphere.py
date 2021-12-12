import taichi as ti

from src.Simulation import Simulation
from src.objects.Sphere import Sphere
from src.objects.Plane import Plane
from src.clothes.GridCloth import GridCloth

sim = Simulation("test", res=(1600, 900), dt=0.01, MODE=ti.cpu, iterations=3)
sim.set_camera((-0.8, 1.6, 0.5), (0, 0, 0))
sim.add_light((-0.1, -0.1, 1), (1, 1, 1))
sim.set_external_force(ti.Vector([0, 8, 0]))

c = GridCloth(N=(32, 32), S=(0.5, 0.5), KS=1  , KB=0.2, KC=1, center=ti.Vector([0, 0, 0.01]))
sim.add_cloth(c)

s = Sphere(ti.Vector([0, 0, -0.11]), 0.1, color=(1, 0, 0))
sim.add_object(s)

p = Plane(ti.Vector([0, 0, -0.3]), color=(0.3, 0.3, 0.3), drest = 0.01)
sim.add_object(p)

#sim.make_video("sim")
sim.run()