import taichi as ti

@ti.kernel
def kernel(p : ti.template()):
    print(1)


a = ti.Vector.field(3, float, (10,))
for i in range(10):
    a[i] = ti.Vector([i, i, i])

kernel(a)