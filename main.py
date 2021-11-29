import taichi as ti

ti.init(arch=ti.cpu)

# ------- PARAMETERS ---------


N = 64
CLOTH_SIZE = 0.5
DELTA_SIZE = CLOTH_SIZE / N

GRAVITY = 4
DAMPING = 0.01
DT = 0.001
STRETCHING_STIFFNESS = 0.2
COLLISION_STIFFNESS = 0.1
RESX, RESY = 1600, 900

p = ti.Vector.field(3, float, (N, N)) # cloth particle predictions
x = ti.Vector.field(3, float, (N, N)) # cloth particle positions
v = ti.Vector.field(3, float, (N, N)) # cloth particle velocities

num_triangles = (N - 1) * (N - 1) * 2
indices = ti.field(int, num_triangles * 3)
vertices = ti.Vector.field(3, float, N * N)

def init_cloth():
    for i, j in ti.ndrange(N, N):
        x[i, j] = ti.Vector([
            i * DELTA_SIZE-CLOTH_SIZE/2, 
            j * DELTA_SIZE-CLOTH_SIZE/2,
            -0.12
        ])
        v[i, j] = ti.Vector([0, 0, 0])

@ti.kernel
def init_triangles():
    for i, j in ti.ndrange(N, N):
        if i < N - 1 and j < N - 1:
            square_id = (i * (N - 1)) + j
            # 1st triangle of the square
            indices[square_id * 6 + 0] = i * N + j
            indices[square_id * 6 + 1] = (i + 1) * N + j
            indices[square_id * 6 + 2] = i * N + (j + 1)
            # 2nd triangle of the square
            indices[square_id * 6 + 3] = (i + 1) * N + j + 1
            indices[square_id * 6 + 4] = i * N + (j + 1)
            indices[square_id * 6 + 5] = (i + 1) * N + j

@ti.kernel
def set_vertices():
    for i, j in ti.ndrange(N, N):
        vertices[i * N + j] = x[i, j]

init_cloth()
init_triangles()
set_vertices()


window = ti.ui.Window("Cloth Simulation (PBD)", (RESX, RESY), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

x_delta = ti.Vector.field(3, float, (N, N))


@ti.func
def solve_constraints(stride):
    # particle corrections
    for i, j in ti.ndrange(N, N):
        x_delta[i, j] = ti.Vector([0, 0, 0])
    
    
    # stretching constraint
    for i, j in ti.ndrange(N, N):
        # right
        if j + stride < N:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i,j+stride]) / (p[i, j] - p[i,j+stride]).norm()
            lagrange = ((p[i, j] - p[i,j+stride]).norm() - d) / 2

            x_delta[i, j]   += -lagrange * n
            x_delta[i, j+stride] +=  lagrange * n
        """
        # diagonal (down right)
        if j + stride < N and i + stride < N:
            d = stride * ti.sqrt(2*(DELTA_SIZE**2))
            n = (p[i, j] - p[i+stride,j+stride]) / (p[i, j] - p[i+stride,j+stride]).norm()
            lagrange = ((p[i, j] - p[i+stride,j+stride]).norm() - d) / 2

            x_delta[i, j]       += -lagrange * n
            x_delta[i+stride, j+stride]   +=  lagrange * n
        """
        # down
        if i + stride < N:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i+stride,j]) / (p[i, j] - p[i+stride,j]).norm()
            lagrange = ((p[i, j] - p[i+stride,j]).norm() - d) / 2

            x_delta[i, j]   += -lagrange * n
            x_delta[i+stride, j] +=  lagrange * n

        # up
        if i - stride >= 0:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i-stride,j]) / (p[i, j] - p[i-stride,j]).norm()
            lagrange = ((p[i, j] - p[i-stride,j]).norm() - d) / 2

            x_delta[i, j]   += -lagrange * n
            x_delta[i-stride,j] +=  lagrange * n
        
        # left
        if j - stride >= 0:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i,j-stride]) / (p[i, j] - p[i,j-stride]).norm()
            lagrange = ((p[i, j] - p[i,j-stride]).norm() - d) / 2

            x_delta[i, j]   += -lagrange * n
            x_delta[i, j-stride] +=  lagrange * n
        """
        # diagonal (up left)
        if j - stride >= 0 and i - stride >= 0:
            d = stride * ti.sqrt(2*(DELTA_SIZE**2))
            n = (p[i, j] - p[i-stride,j-stride]) / (p[i, j] - p[i-stride,j-stride]).norm()
            lagrange = ((p[i, j] - p[i-stride,j-stride]).norm() - d) / 2

            x_delta[i, j]       += -lagrange * n
            x_delta[i-stride, j-stride]   +=  lagrange * n

        # diagonal (up right)
        if j + stride < N and i - stride >= 0:
            d = stride * ti.sqrt(2*(DELTA_SIZE**2))
            n = (p[i, j] - p[i-stride,j+stride]) / (p[i, j] - p[i-stride,j+stride]).norm()
            lagrange = ((p[i, j] - p[i-stride,j+stride]).norm() - d) / 2

            x_delta[i, j]       += -lagrange * n
            x_delta[i-stride, j+stride]   +=  lagrange * n
        
        # diagonal (down left)
        if j - stride >= 0 and i + stride < N:
            d = stride * ti.sqrt(2*(DELTA_SIZE**2))
            n = (p[i, j] - p[i+stride,j-stride]) / (p[i, j] - p[i+stride,j-stride]).norm()
            lagrange = ((p[i, j] - p[i+stride,j-stride]).norm() - d) / 2

            x_delta[i, j]       += -lagrange * n
            x_delta[i+stride, j-stride]   +=  lagrange * n
        """
    for i, j in ti.ndrange(N, N):
        p[i, j] += x_delta[i, j] * STRETCHING_STIFFNESS

    # Environment Collision Constraint
    DREST = 0.1
    for i, j in ti.ndrange(N, N):
        if p[i,j].z > 0. - DREST and p[i,j].x < 0.2 and p[i,j].x > -0.2 and p[i,j].y < 0.2 and p[i,j].y > -0.2:
            n = ti.Vector([0, 0, -1.0])
            x_delta[i,j] += - (n.dot(p[i,j]) - DREST) * n


    for i, j in ti.ndrange(N, N):
        p[i, j] += x_delta[i, j] * COLLISION_STIFFNESS
         

@ti.kernel
def step():
    # Gravity, external forces
    for i in ti.grouped(v):
        v[i].z += DT * GRAVITY

    # make predictions for the positions
    for i in ti.grouped(x):
        p[i] = x[i] + DT * v[i]  

    # make and solve constraints

    for _ in range(1):
        for i in range(1):
            solve_constraints(1)
            #solve_constraints(2)
            #solve_constraints(4)
            #solve_constraints(8)
            #solve_constraints(16)
            #solve_constraints(32)
            #solve_constraints(16)
            #solve_constraints(8)
            #solve_constraints(4)
            #solve_constraints(2)
            solve_constraints(1)
            
        
        

    # set obtained positions and update velocities accordingly
    for i in ti.grouped(x):
        v[i] = (x[i] - p[i]) / DT * DAMPING
        x[i] = p[i]


# orientation
"""
origin = ti.Vector.field(3, float, (1, ))
or_x = ti.Vector.field(3, float, (1, ))
or_y = ti.Vector.field(3, float, (1, ))
or_z = ti.Vector.field(3, float, (1, ))

origin[0] = ti.Vector([0, 0, 0])
or_x[0] = ti.Vector([0.1, 0, 0])
or_y[0] = ti.Vector([0, 0.1, 0])
or_z[0] = ti.Vector([0, 0, 0.1])

scene.particles(origin, radius=0.05, color=(1, 1, 1)) # origin
scene.particles(or_x, radius=0.05, color=(1, 0, 0)) # x
scene.particles(or_y, radius=0.05, color=(0, 1, 0)) # y
scene.particles(or_z, radius=0.05, color=(0, 0, 1)) # z
"""
while window.running:

    #solver here
    step()
    set_vertices()

    camera.position(0, -1, -0.3)
    camera.lookat(0, 0, -0.1)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))

    scene.mesh(vertices,
               indices=indices,
               color=(0.5, 0.5, 0.5),
               two_sided=True)
    #scene.particles(ball_center, radius=ball_radius, color=(0.5, 0, 0))
    canvas.scene(scene)
    window.show()