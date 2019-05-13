# PSO # 

[Particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality.

Given an objective function, two equations determine the algorithms

 1. V(t+1) = w * V(t) + r1 * c1* (P - X) + r2* c2 * (G - X)
 2. X(t+1) = X(t) + V(t)
 
 Where:
  * V contains velocity of particles
  * X contains positions of particles
  * w, r1, r2, c1, and c2 are constants
  * P contains the best known position of each particle
  * G contains the best known position ever known


## Run
Open a terminal and type:

```bash
sh run.sh
```

## Output ##
```
Total particles number: 20
Step 0 (w=0.73) :: min err=1.79966e+03
Goal achieved @ step 83 (error=3.002e-06) :-)
Best known position: [  0.00  0.00  0.00 -0.00  0.00  0.00 -0.00  0.00  0.00 -0.00  0.00 -0.00  0.00 -0.00  0.00 -0.00  0.00  0.00 -0.00 -0.00  0.00 -0.00 -0.00 -0.00 -0.00  0.00  0.00  0.00 -0.00 -0.00]
```

## References ##

 * [Particle Swarm Optimization (PSO) in C](https://github.com/kkentzo/pso)
