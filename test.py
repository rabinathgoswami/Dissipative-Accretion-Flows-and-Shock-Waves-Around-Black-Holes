from parameters import p
from functions import master_critical_point_solver_eps


# a,b,c= master_critical_point_solver_eps(p, 3.107, 0.005)

z,a,b,c,d,e,f,g = master_critical_point_solver_eps(p, 3.107, 0.005)

print(z, a,b,c,d,e,f,g)