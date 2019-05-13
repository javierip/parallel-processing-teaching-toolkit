#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>

#include "pso.h"
#include "objectives.h"

int main()
{
    //initialize random number generator
    srand(time(NULL));

    // define objective function
    pso_obj_fun_t obj_fun = pso_sphere;

    // initialize pso settings
    pso_settings_t settings;

    // set the default settings
    pso_set_default_settings(&settings);

    // initialize GBEST solution
    pso_result_t solution;
    // allocate memory for the best position buffer
    solution.gbest = (double *)malloc(settings.dim * sizeof(double));

    // run optimization algorithm
    pso_solve(obj_fun, NULL, &solution, &settings);

    // free the gbest buffer
    free(solution.gbest);


    return 0;
}
