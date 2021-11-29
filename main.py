import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# ------- PARAMETERS ---------

# Solver Properties
GRAVITY = 10                 # gravity
DAMPING = 0.8               # 0-1 where 1 = no damping
SOLVER_ITERATIONS = 2       # Iterations for solver optimization
DT = 0.001                  # Time delta

# Material Properties
N = 20                      # NxN grid
CLOTH_SIZE = 0.5            # Size in 3d coords
DELTA_SIZE = CLOTH_SIZE / N # space between 2 grid points

STRETCHING_STIFFNESS = 0.9 # 0-1 where 1 = stiff
WD = np.pi                  # initial angle between triangles
BENDING_STIFFNESS = 0.01     # 0-1 where 1 = no bending
COLLISION_STIFFNESS = 0.31  # 0-1 where 1 = max repulsion

#Window Properties
RESX, RESY = 1600, 900      # window resolution

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
            -0.010001
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
def solve_stretching(stride):
    KS = (1 - STRETCHING_STIFFNESS)**SOLVER_ITERATIONS # stretching coeff

    # stretching constraint
    for i, j in ti.ndrange(N, N):
        # right
        if j + stride < N:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i,j+stride]) / (p[i, j] - p[i,j+stride]).norm()
            lagrange = ((p[i, j] - p[i,j+stride]).norm() - d) / 2

            x_delta[i, j]   += -KS * lagrange * n
            x_delta[i, j+stride] += KS * lagrange * n
        
        # diagonal (down right)
        if j + stride < N and i + stride < N:
            d =  ti.sqrt(2*((DELTA_SIZE * stride)**2))
            n = (p[i, j] - p[i+stride,j+stride]) / (p[i, j] - p[i+stride,j+stride]).norm()
            lagrange = ((p[i, j] - p[i+stride,j+stride]).norm() - d) / 2

            x_delta[i, j]       += - KS * lagrange * n
            x_delta[i+stride, j+stride]   +=  KS * lagrange * n
        
        # down
        if i + stride < N:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i+stride,j]) / (p[i, j] - p[i+stride,j]).norm()
            lagrange = ((p[i, j] - p[i+stride,j]).norm() - d) / 2

            x_delta[i, j]   += -KS * lagrange * n
            x_delta[i+stride, j] += KS * lagrange * n
        """
        # up
        if i - stride >= 0:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i-stride,j]) / (p[i, j] - p[i-stride,j]).norm()
            lagrange = ((p[i, j] - p[i-stride,j]).norm() - d) / 2

            x_delta[i, j]   += - KS * lagrange * n
            x_delta[i-stride,j] +=  KS * lagrange * n
        
        # left
        if j - stride >= 0:
            d = stride * DELTA_SIZE
            n = (p[i, j] - p[i,j-stride]) / (p[i, j] - p[i,j-stride]).norm()
            lagrange = ((p[i, j] - p[i,j-stride]).norm() - d) / 2

            x_delta[i, j]   += -lagrange * n
            x_delta[i, j-stride] +=  lagrange * n
        
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
@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)

@ti.func
def get_bending_correction(p1, p2, p3, p4):
    op2 = p2 - p1
    op3 = p3 - p1
    op4 = p4 - p1

    n1 = op2.cross(op3)
    n1 = n1 / n1.norm()

    n2 = op2.cross(op4)
    n2 = n2 / n2.norm()

    d = n1.dot(n2)

    while d < -1:
        d += 2
    while d > 1:
        d -= 2

    q3 = (op2.cross(n2) + n1.cross(op2) * d) / (op2.cross(op3)).norm() 
    q4 = (op2.cross(n1) + n2.cross(op2) * d) / (op2.cross(op4)).norm()
    q2 = (-(op3.cross(n2) + n1.cross(op3) * d) / (op2.cross(op3)).norm()) - (op4.cross(n1) + n2.cross(op4) * d) / (op2.cross(op4)).norm()
    q1 = -q2 - q3 - q4

    

    

    sd = -ti.sqrt(1 - d*d) * (ti.acos(d) - WD) 

    S = q1.norm()**2 + q2.norm()**2 + q3.norm()**2 + q4.norm()**2
    
    delta_p1, delta_p2, delta_p3, delta_p4 = ti.Vector([0, 0, 0]), ti.Vector([0, 0, 0]), ti.Vector([0, 0, 0]), ti.Vector([0, 0, 0])

    if not sd == 0:
        delta_p1 = (sd/S * q1)
        delta_p2 = (sd/S * q2)
        delta_p3 = (sd/S * q3)
        delta_p4 = (sd/S * q4)

    return delta_p1, delta_p2, delta_p3, delta_p4

@ti.func
def solve_bending(stride):
    KB = (BENDING_STIFFNESS)**SOLVER_ITERATIONS


    # diag
    for i, j in ti.ndrange(N, N):
        if i+stride < N and j + stride < N:
            d1, d2, d3, d4 = get_bending_correction(p[i, j], p[i+1, j+1], p[i, j+1], p[i+1,j])
            
            x_delta[i, i] += KB * d1
            x_delta[i+1, j+1] += KB * d2
            x_delta[i, j+1] += KB * d3
            x_delta[i+1, j] += KB * d4
    
    # left
    for i, j in ti.ndrange(N, N):
        if i-stride >= 0 and i+stride < N and j + stride < N:
            d1, d2, d3, d4 = get_bending_correction(p[i, j], p[i, j+1], p[i-1, j], p[i+1,j+1])
            
            x_delta[i, i] += KB * d1
            x_delta[i, j+1] += KB * d2
            x_delta[i-1, j] += KB * d3
            x_delta[i+1, j+1] += KB * d4

    # down
    for i, j in ti.ndrange(N, N):
        if j-stride >= 0 and i+stride < N and j + stride < N:
            d1, d2, d3, d4 = get_bending_correction(p[i, j], p[i+1, j], p[i+1, j+1], p[i,j-1])
            
            x_delta[i, i] += KB * d1
            x_delta[i+1, j] += KB * d2
            x_delta[i+1, j+1] += KB * d3
            x_delta[i, j-1] += KB * d4
            
@ti.func
def solve_environment_collison():
    DREST = 0.01
    KC = COLLISION_STIFFNESS ** SOLVER_ITERATIONS

    for i, j in ti.ndrange(N, N):
        if p[i,j].z < 0. + DREST and p[i,j].z > 0. - DREST and p[i,j].x < 0.2 and p[i,j].x > -0.2 and p[i,j].y < 0.2 and p[i,j].y > -0.2:
            n = ti.Vector([0, 0, -1.0])
            x_delta[i,j] += - KC * (n.dot(p[i,j]) - DREST) * n

@ti.func
def solve_constraints(stride):
    # particle corrections
    for i, j in ti.ndrange(N, N):
        x_delta[i, j] = ti.Vector([0, 0, 0])
    
    solve_stretching(stride)
    solve_bending(stride)
    solve_environment_collison()

    for i, j in ti.ndrange(N, N):
        p[i, j] += x_delta[i, j]

         

@ti.kernel
def step():
    # Gravity, external forces
    for i in ti.grouped(v):
        v[i].z += DAMPING * DT * GRAVITY

    # make predictions for the positions
    for i in ti.grouped(x):
        p[i] = x[i] + DT * v[i]  

    # make and solve constraints

    for _ in range(1):
        for i in range(SOLVER_ITERATIONS):
            solve_constraints(1)
            solve_constraints(10)
        
        

    # set obtained positions and update velocities accordingly
    for i in ti.grouped(x):
        v[i] = (x[i] - p[i]) / DT
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