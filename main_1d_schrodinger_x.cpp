//#define DEBUG

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <time.h>
#include "Mesh.hpp"
#include "Coords.hpp"
#include "Hierarchy.hpp"
#include "routines.hpp"

int main() {
   int num_l;
   int num_f;
   int dim;
   int i, j, n;
   int size_base[3];
   double bbox[6];
   double *sigma1_n, *sigma2_n, *sigma1_np1, *sigma2_np1;
   double* x, dx; 

   num_l = 11;
   num_f = 4;
   dim = 1;
   size_base[0] = 3;
   size_base[1] = 3;
   size_base[2] = 3;
   bbox[0] = -1;
   bbox[1] = 1;
   bbox[2] = -1;
   bbox[3] = 1;
   bbox[4] = -1;
   bbox[5] = 1;

   Hierarchy* hierarchy = new Hierarchy(num_l, num_f, dim, size_base, bbox);

   n = hierarchy->fields[num_l - 1][0]->n;
   sigma1_n = hierarchy->fields[num_l - 1][0]->f1d;
   sigma2_n = hierarchy->fields[num_l - 1][1]->f1d;
   sigma1_np1 = hierarchy->fields[num_l - 1][2]->f1d;
   sigma2_np1 = hierarchy->fields[num_l - 1][3]->f1d;
   x = hierarchy->coords[num_l - 1]->mesh[0]->f1d;
   dx = x[1] - x[0];
   for (i = 0; i < n; i++) {
      sigma1_n[i] = -sin(x[i]*M_PI/2.0-M_PI/2.0);
      sigma2_n[i] = 0;
      sigma1_np1[i] = -sin(x[i]*M_PI/2.0-M_PI/2.0);
      sigma2_np1[i] = 0;
   }


   double begin, end;
   double time_spent;

   //omp_set_num_threads(2);
   //printf("begin\n");
   //begin = omp_get_wtime();
   
   //hierarchy->relaxLevel(4, 1, num_l-1, f_schrodinger_opp_1d, f_schrodinger_res_1d, f_schrodinger_jac_1d);

   double t = 0.1;
   for (j = 0; j < t/(0.1*dx); j++)
   {
      printf("step:\n\n");
      for (i = 0; i < 20; i++) {
         hierarchy->vCycle(4, dx*0.1, f_schrodinger_opp_1d, 
                 f_schrodinger_jac_1d);
         printf("change %e, residual %e\n", hierarchy->change[num_l - 1], 
                 hierarchy->resid[num_l - 1]);
      }
      hierarchy->switchFields(0,2);
      hierarchy->switchFields(1,3);
   }

   hierarchy->switchFields(0,2);
   hierarchy->switchFields(1,3);
   
   //printf("after:\n");
   //end = omp_get_wtime();
   //time_spent = (end - begin);
   //printf("time spent: %lf\n", time_spent);
   
   //hierarchy->print(num_l-1);
   f_schrodinger_indep_resid_1d(dx*0.1, hierarchy->coords[num_l-1], hierarchy->fields[num_l-1]);

   delete hierarchy;

   return 0;
}
