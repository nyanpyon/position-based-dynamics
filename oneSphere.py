import taichi as ti

from src.Simulation import Simulation
from src.objects.Box import Box
from src.objects.Sphere import Sphere
from src.clothes.GridCloth import GridCloth

sim = Simulation("test", res=(1600, 900), dt=0.01,MODE=ti.cpu)
sim.set_camera((-0.8, 1.6, 0.5), (0, 0, 0))
sim.add_light((-0.1, -0.1, 1), (1, 1, 1))

c = GridCloth(N=(32, 32), S=(0.5, 0.5), KS=1  , KB=0.2, KC=1, center=ti.Vector([0, 0, 0.01]))
sim.add_cloth(c)

s = Sphere(ti.Vector([0, 0, -0.11]), 0.1, color=(1, 0, 0))
sim.add_object(s)

#sim.make_video("sim3-4-5")
sim.run()