/*
 * This program draws Mandelbrot set for Fc(z)=z*z +c using Mandelbrot algorithm ( boolean escape time )
 * It is based on the source provided by Rosetta Code (https://rosettacode.org/wiki/Mandelbrot_set#PPM_non_interactive)
 * Tue output file is output.ppm
 * If you want to convert it: convert -normalize output.ppm output.png
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

typedef struct{
    unsigned char value [3];
}pixel;

const int MAX_COLOR_COMPONENT_VALUE = 255;

void computeSet(pixel* image, const int MAX_X_POINTS, const int MAX_Y_POINTS);
void saveData(const pixel* image, const int MAX_X_POINTS, const int MAX_Y_POINTS);

int main()
{
    const int MAX_X_POINTS = 800;
    const int MAX_Y_POINTS = 800;

    pixel * image;
    image = (pixel *) malloc(MAX_Y_POINTS * MAX_X_POINTS * sizeof(pixel));

    computeSet(image, MAX_X_POINTS, MAX_Y_POINTS);
    saveData(image, MAX_X_POINTS, MAX_Y_POINTS);

    free(image);
    image = NULL;

    return 0;
}

void computeSet(pixel* image, const int MAX_X_POINTS, const int MAX_Y_POINTS)
{
    int actualX,actualY;

    double cooddinateX, coordinateY;
    const double COORDINATE_X_MIN = -2.5;
    const double COORDINATE_X_MAX = 1.5;
    const double COORDINATE_Y_MIN = -2.0;
    const double COORDINATE_Y_MAX = 2.0;

    /* Calclulate pixels */
    double pixelWidth = (COORDINATE_X_MAX - COORDINATE_X_MIN) / MAX_X_POINTS;
    double pixelHeight = (COORDINATE_Y_MAX - COORDINATE_Y_MIN) / MAX_Y_POINTS;

    /* Z=Zx+Zy*i  ;   Z0 = 0 */
    double Zx, Zy;
    double Zx2, Zy2; /* Zx2=Zx*Zx;  Zy2=Zy*Zy  */
    /*  */
    int iteration;
    const int MAX_ITERATION_NUMBER = 200;
    /* bail-out value , radius of circle ;  */
    const double ESCAPE_RADIUS = 2;
    double ER_2 = ESCAPE_RADIUS*ESCAPE_RADIUS;

    /* compute and write image data bytes to the file*/
    for(actualY=0; actualY < MAX_Y_POINTS; actualY++)
    {
        coordinateY = COORDINATE_Y_MIN + actualY * pixelHeight;
        if (fabs(coordinateY)< pixelHeight/2) coordinateY=0.0; /* Main antenna */
        for(actualX=0; actualX < MAX_X_POINTS; actualX++)
        {
            cooddinateX = COORDINATE_X_MIN + actualX*pixelWidth;
            /* initial value of orbit = critical point Z= 0 */
            Zx=0.0;
            Zy=0.0;
            Zx2=Zx*Zx;
            Zy2=Zy*Zy;
            /* */
            for (iteration=0;iteration<MAX_ITERATION_NUMBER && ((Zx2+Zy2)<ER_2);iteration++)
            {
                Zy=2*Zx*Zy + coordinateY;
                Zx=Zx2-Zy2 +cooddinateX;
                Zx2=Zx*Zx;
                Zy2=Zy*Zy;
            };
            /* compute  pixel color (24 bit = 3 bytes) */
            if (iteration==MAX_ITERATION_NUMBER)
            { /*  interior of Mandelbrot set = black */

                image[actualX + actualY * MAX_X_POINTS].value[0] = 0;
                image[actualX + actualY * MAX_X_POINTS].value[1] = 0;
                image[actualX + actualY * MAX_X_POINTS].value[2] = 0;
            }
            else
            { /* exterior of Mandelbrot set = white */
                image[actualX + actualY * MAX_X_POINTS].value[0] = MAX_COLOR_COMPONENT_VALUE;
                image[actualX + actualY * MAX_X_POINTS].value[1] = MAX_COLOR_COMPONENT_VALUE;
                image[actualX + actualY * MAX_X_POINTS].value[2] = MAX_COLOR_COMPONENT_VALUE;
            };
        }
    }
}

void saveData(const pixel* image, const int MAX_X_POINTS, const int MAX_Y_POINTS)
{
    int actualX,actualY;
    /* it is 24 bit color RGB file */

    FILE * fp;
    char *filename = "output.ppm";
    char *comment="# ";/* comment should start with # */

    /*create new file,give it a name and open it in binary mode  */
    fp= fopen(filename,"wb"); /* b -  binary mode */
    /*write ASCII header to the file*/
    fprintf(fp,"P6\n %s\n %d\n %d\n %d\n",comment, MAX_X_POINTS, MAX_Y_POINTS, MAX_COLOR_COMPONENT_VALUE);

    for(actualY=0;actualY<MAX_Y_POINTS;actualY++)
    {
        for(actualX=0;actualX<MAX_X_POINTS;actualX++)
        {
            fwrite(image[actualX + actualY * MAX_X_POINTS].value, 1, 3, fp);
        }
    }
    fclose(fp);
}
