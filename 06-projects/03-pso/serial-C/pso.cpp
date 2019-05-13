#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "pso.h"

// calulate swarm size based on dimensionality
int pso_calc_swarm_size(int dim) {
    int size = 10. + 2. * sqrt(dim);
    return (size > PSO_MAX_SIZE ? PSO_MAX_SIZE : size);
}

// return default pso settings
void pso_set_default_settings(pso_settings_t *settings) {

    // set some default values
    settings->dim = 30;
    settings->x_lo = -20;
    settings->x_hi = 20;
    settings->goal = 1e-5;

    settings->size = pso_calc_swarm_size(settings->dim);
    settings->print_every = 1000;
    settings->steps = 100000;
    settings->c1 = 1.496;
    settings->c2 = 1.496;
    settings->w_max = PSO_INERTIA;
    settings->w_min = 0.3;

    settings->clamp_pos = 1;
}

float random_float(float min, float max)
{
    float random = ((float) rand()) / (float) RAND_MAX;
    float dif = max - min;
    float range = random * dif;
    return min + range;
}

void pso_solve(pso_obj_fun_t obj_fun, void *obj_fun_params,pso_result_t *solution, pso_settings_t *settings)
{
    printf("Total particles number: %d\n", settings->size);

    // Particles
    double pos[settings->size][settings->dim]; // position matrix
    double vel[settings->size][settings->dim]; // velocity matrix
    double pos_b[settings->size][settings->dim]; // best position matrix
    double fit[settings->size]; // particle fitness vector
    double fit_b[settings->size]; // best fitness vector

    // Swarm
    double pos_nb[settings->size][settings->dim]; // what is the best informed

    // position for each particle
    int comm[settings->size][settings->size]; // communications:who informs who
    int improved; // whether solution->error was improved duringthe last iteration

    int part_id, dim_id, step;
    double a, b; // for matrix initialization
    double rho1, rho2; // random numbers (coefficients)
    double w; // current omega
    void (*inform_fun)(); // neighborhood update function
    double (*calc_inertia_fun)(); // inertia weight update function


    // INITIALIZE SOLUTION
    solution->error = DBL_MAX;

    // SWARM INITIALIZATION
    // for each particle
    for (part_id=0; part_id<settings->size; part_id++) {
        // for each dimension
        for (dim_id=0; dim_id<settings->dim; dim_id++) {
            // generate two numbers within the specified range
            a = settings->x_lo + (settings->x_hi - settings->x_lo) * random_float(0,1);
            b = settings->x_lo + (settings->x_hi - settings->x_lo) * random_float(0,1);
            // initialize position
            pos[part_id][dim_id] = a;
            // best position is the same
            pos_b[part_id][dim_id] = a;
            // initialize velocity
            vel[part_id][dim_id] = (a-b) / 2.;
        }
        // update particle fitness
        fit[part_id] = obj_fun(pos[part_id], settings->dim, obj_fun_params);
        fit_b[part_id] = fit[part_id]; // this is also the personal best
        // update gbest??
        if (fit[part_id] < solution->error) {
            // update best fitness
            solution->error = fit[part_id];
            // copy particle pos to gbest vector
            memmove((void *)solution->gbest, (void *)&pos[part_id],sizeof(double) * settings->dim);
        }
    }

    // initialize omega using standard value
    w = PSO_INERTIA;
    // RUN ALGORITHM
    for (step=0; step<settings->steps; step++) {
        // update current step
        settings->step = step;

        // check optimization goal
        if (solution->error <= settings->goal)
        {
            // SOLVED!!
            if (settings->print_every)
                printf("Goal achieved @ step %d (error=%.3e) :-)\n", step, solution->error);
            printf("Best known position: [");
            for (part_id=0; part_id<settings->dim; part_id++) {
                printf("%6.2lf", solution->gbest[part_id]);
            }
            printf("]\n");
            break;
        }

        // the value of improved was just used; reset it
        improved = 0;

        // update all particles
        for (part_id=0; part_id<settings->size; part_id++) {
            // for each dimension
            for (dim_id=0; dim_id<settings->dim; dim_id++) {
                // calculate stochastic coefficients
                rho1 = settings->c1 * random_float(0,1);
                rho2 = settings->c2 * random_float(0,1);
                // update velocity
                vel[part_id][dim_id] = w * vel[part_id][dim_id] +	\
                        rho1 * (pos_b[part_id][dim_id] - pos[part_id][dim_id]) +	\
                        rho2 * (pos_nb[part_id][dim_id] - pos[part_id][dim_id]);
                // update position
                pos[part_id][dim_id] += vel[part_id][dim_id];
                // clamp position within bounds?
                if (settings->clamp_pos) {
                    if (pos[part_id][dim_id] < settings->x_lo) {
                        pos[part_id][dim_id] = settings->x_lo;
                        vel[part_id][dim_id] = 0;
                    } else if (pos[part_id][dim_id] > settings->x_hi) {
                        pos[part_id][dim_id] = settings->x_hi;
                        vel[part_id][dim_id] = 0;
                    }
                } else {
                    // enforce periodic boundary conditions
                    if (pos[part_id][dim_id] < settings->x_lo) {

                        pos[part_id][dim_id] = settings->x_hi - fmod(settings->x_lo - pos[part_id][dim_id],
                                                          settings->x_hi - settings->x_lo);
                        vel[part_id][dim_id] = 0;

                    } else if (pos[part_id][dim_id] > settings->x_hi) {

                        pos[part_id][dim_id] = settings->x_lo + fmod(pos[part_id][dim_id] - settings->x_hi,
                                                          settings->x_hi - settings->x_lo);
                        vel[part_id][dim_id] = 0;
                    }
                }

            }

            // update particle fitness
            fit[part_id] = obj_fun(pos[part_id], settings->dim, obj_fun_params);
            // update personal best position?
            if (fit[part_id] < fit_b[part_id]) {
                fit_b[part_id] = fit[part_id];
                // copy contents of pos[i] to pos_b[i]
                memmove((void *)&pos_b[part_id], (void *)&pos[part_id],sizeof(double) * settings->dim);
            }
            // update gbest??
            if (fit[part_id] < solution->error) {
                improved = 1;
                // update best fitness
                solution->error = fit[part_id];
                // copy particle pos to gbest vector
                memmove((void *)solution->gbest, (void *)&pos[part_id],sizeof(double) * settings->dim);
            }
        }

        if (settings->print_every && (step % settings->print_every == 0))
            printf("Step %d (w=%.2f) :: min err=%.5e\n", step, w, solution->error);

    }
}


