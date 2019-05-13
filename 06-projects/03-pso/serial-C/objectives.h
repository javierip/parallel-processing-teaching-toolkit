#ifndef OBJECTIVES_H
#define OBJECTIVES_H

double pso_sphere(double *vec, int dim, void *params) {
    double sum = 0;
    int i;
    for (i=0; i<dim; i++)
        sum += pow(vec[i], 2);

    return sum;
}

double pso_rosenbrock(double *vec, int dim, void *params) {
  double sum = 0;
  int i;
  for (i=0; i<dim-1; i++)
    sum += 100 * pow((vec[i+1] - pow(vec[i], 2)), 2) +	\
      pow((1 - vec[i]), 2);

  return sum;
}



#endif // OBJECTIVES_H
