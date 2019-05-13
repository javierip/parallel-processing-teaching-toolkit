#ifndef PSO_H
#define PSO_H


// PSO SETTINGS
typedef struct {

    int dim; // problem dimensionality
    double x_lo; // lower range limit
    double x_hi; // higher range limit
    double goal; // optimization goal (error threshold)

    int size; // swarm size (number of particles)
    int print_every; // ... N steps (set to 0 for no output)
    int steps; // maximum number of iterations
    int step; // current PSO step
    double c1; // cognitive coefficient
    double c2; // social coefficient
    double w_max; // max inertia weight value
    double w_min; // min inertia weight value

    int clamp_pos; // whether to keep particle position within defined bounds (TRUE)
    // or apply periodic boundary conditions (FALSE)

} pso_settings_t;

// PSO SOLUTION -- Initialized by the user
typedef struct {
    double error;
    double *gbest; // should contain DIM elements!!

} pso_result_t;

// OBJECTIVE FUNCTION TYPE
typedef double (*pso_obj_fun_t)(double *, int, void *);

// CONSTANTS
#define PSO_MAX_SIZE 100 // max swarm size
#define PSO_INERTIA 0.7298 // default value of w (see clerc02)

void pso_set_default_settings(pso_settings_t *settings);
void pso_solve(pso_obj_fun_t obj_fun, void *obj_fun_params,pso_result_t *solution, pso_settings_t *settings);

#endif // PSO_H
