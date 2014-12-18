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
   Mesh* phi;

   num_l = 8;
   num_f = 1;
   dim = 3;
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
   phi = hierarchy->fields[num_l - 1][0];
   for (i = 0; i < n; i++) {
      phi->data[i] = 0;
   }


   double begin, end;
   double time_spent;
   printf("begin\n");
#ifdef OMP_PARALLEL
   omp_set_num_threads(4);
   begin = omp_get_wtime();
#else
   clock_t begin_clock, end_clock;
   begin_clock = clock();
   begin = (double) begin_clock/CLOCKS_PER_SEC;
#endif 

   for (i = 0; i < 20; i++) {
      hierarchy->vCycle(4, 0, f_poisson_opp_3d, f_poisson_jac_3d);
      printf("change %e, residual %e\n ", hierarchy->change[num_l - 1], 
              hierarchy->resid[num_l - 1]);
   }

#ifdef OMP_PARALLEL
   end = omp_get_wtime();
#else
   end_clock = clock();
   end = (double) end_clock/CLOCKS_PER_SEC;
#endif
   time_spent = (end - begin);
   printf("time spent: %lf\n", time_spent);
   
   //hierarchy->relaxLevel(10000,0,num_l-1,f_poisson_opp_3d, f_poisson_jac_3d);
   //hierarchy->print(num_l-1);
   f_poisson_indep_resid_3d(0.0, hierarchy->coords[num_l-1], hierarchy->fields[num_l-1]);

   delete hierarchy;

   return 0;
}
