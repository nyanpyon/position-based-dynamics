from typing import Mapping
import taichi as ti
import numpy as np
import math
from src.objects.Box import Object 
from src.clothes.Cloth import Cloth 
import os
import imageio
from tqdm import tqdm

class Simulation():
    """
    new simulation
        name: string
            name of the simulation
        gravity: float
            gravity < 0
        dt: float
            time step size used in the simulation
        res: tuple(width, height)
            window size in pixels
    """
    def __init__(self, name, gravity=-10, dt=0.005, res=(500, 500), iterations=4, MODE=ti.cpu):
        ti.init(arch=MODE)

        # simulation properties
        self.name = name
        self.GRAVITY = gravity
        self.DT = dt
        self.NUM_ITERATIONS = iterations
        
        # video manager
        self.video = False
        self.MAX_TIME = -1
        self.frame_rate = 0
        self.temp_path = 'temp/'

        # window properties
        self.RES=res
        self.window = ti.ui.Window(self.name, self.RES, vsync=True)

        # camera
        self.camera_position = ti.Vector([0, -1, 0])
        self.camera_lookat = ti.Vector([0, 0, 0])
        self.camera_up = ti.Vector([0, 0, 1])

        # objects of the scene
        self.lights = []
        self.objects = []
        self.clothes = []

        self.canvas = self.window.get_canvas()
        self.scene  = ti.ui.Scene()
        self.camera = ti.ui.make_camera()

    """
    sets camera position, look at and up
    
    camera_position : tuple(x, y, z)
        position of the camera
    camera_lookat : tuple(x, y, z)
        the point where the camera looks at
    camera_up : tuple(x, y, z)
        the direction where the camera considers as up
    """
    def set_camera(self, camera_position, camera_lookat, camera_up=(0, 0, 1)):
        self.camera_position = camera_position
        self.camera_lookat = camera_lookat
        self.camera_up = camera_up

    def draw_camera(self):
        #self.camera.position(self.camera_position[0], self.camera_position[1], self.camera_position[2])
        self.camera.lookat(self.camera_lookat[0], self.camera_lookat[1], self.camera_lookat[2])
        self.camera.up(self.camera_up[0], self.camera_up[1], self.camera_up[2])   
        self.scene.set_camera(self.camera)

    """
    add new light to the scene
        light_pos : tuple(x, y, z)
            position of the light
        light_color : tuple(r, g, b) 
            color of the light, in range [0, 1]
    """
    def add_light(self, light_pos, light_color):
        self.lights.append((light_pos, light_color))


    """
    adds object to the scene
        obj: Object
            scene object Object.py
    """
    def add_object(self, obj):
        self.objects.append(obj)

    def make_video(self, name, framerate=30, MAX_TIME=-1):
        self.MAX_TIME = MAX_TIME
        self.video = True
        self.frame_rate = framerate
        self.video_path = f"videos/{name}"


    """
    adds object to the scene
        cloth: CLoth
            cloth CLoth.py
    """
    def add_cloth(self, cloth):
        self.clothes.append(cloth)

    """
    CORE PBD algorithm
    """
    def update_cloth(self, c):
        c.external_forces(self.GRAVITY, self.DT)

        c.make_predictions(self.DT)

        #call solver
        for i in range(self.NUM_ITERATIONS):
            c.solve_stretching_constraint(self.NUM_ITERATIONS)
            c.solve_bending_constraints(self.NUM_ITERATIONS)
        
        c.solve_self_collision_constraints()

        for o in self.objects:
            c.solve_collision_constraints(o)

        
            
        c.apply_correction(self.DT)

    """ 
    runs the simulation
    """
    
    
    def run(self):
        TIME = 0
        img_num = 0
        next_frame = 0
        while (self.MAX_TIME == -1 and self.window.running) or TIME < self.MAX_TIME:
            TIME += self.DT
            next_frame += self.DT

            self.camera.position(2 * math.cos(TIME), 2 * math.sin(TIME), 1)
            self.draw_camera()
            

            # add lights
            for l in self.lights:
                self.scene.point_light(pos=l[0], color=l[1])

            for o in self.objects:
                o.draw(self.scene)

            for c in self.clothes:
                self.update_cloth(c)

                c.draw(self.scene)

            self.canvas.scene(self.scene)
            
            if self.video:
                if next_frame >= 10.0/self.frame_rate:
                    self.window.write_image(f"{self.temp_path}image_{img_num:07d}.png")
                    img_num += 1
                    next_frame = 0
            
            self.window.show()
        if self.video:
            images = []
            for f in os.listdir(self.temp_path):
                filename = os.path.join(self.temp_path, f)
                images.append(filename)
                
            with imageio.get_writer(f"{self.video_path}.gif", mode='I', fps=self.frame_rate) as writer:
                for filename in tqdm(images):
                    image = imageio.imread(filename)
                    writer.append_data(image)
                    os.remove(filename)




