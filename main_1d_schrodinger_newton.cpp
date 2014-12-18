//#define DEBUG
//#define OMP_PARALLEL

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
   int i, j, k, n;
   int size_base[1];
   double bbox[2];
   double *s1_n, *s2_n, *V_n, *s1_np1, *s2_np1, *V_np1;
   double* x, dx; 

   num_l = 9;
   num_f = 6;
   dim = 1;
   size_base[0] = 3;
   bbox[0] = 0;
   bbox[1] = 10;

   Hierarchy* hierarchy = new Hierarchy(num_l, num_f, dim, size_base, bbox);

   n = hierarchy->fields[num_l - 1][0]->n;
   s1_n   = hierarchy->fields[num_l - 1][0]->f1d;
   s2_n   = hierarchy->fields[num_l - 1][1]->f1d;
   V_n    = hierarchy->fields[num_l - 1][2]->f1d;
   s1_np1 = hierarchy->fields[num_l - 1][3]->f1d;
   s2_np1 = hierarchy->fields[num_l - 1][4]->f1d;
   V_np1  = hierarchy->fields[num_l - 1][5]->f1d;
   x = hierarchy->coords[num_l - 1]->mesh[0]->f1d;
   dx = x[1] - x[0];
   for (i = 0; i < n; i++) {
      s1_n[i]   = 10.0*exp(-x[i]*x[i]);
      s1_np1[i] = 10.0*exp(-x[i]*x[i]);
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
   
   double resid = 1;
   while(resid > 1e-8) {
      hierarchy->vCycle(4, 0.0, f_schrodinger_newton_opp_noPsi_1d, 
              f_schrodinger_newton_jac_noPsi_1d);
      printf("change %e, residual %e\n", hierarchy->change[num_l - 1], 
              hierarchy->resid[num_l - 1]);
      resid = hierarchy->resid[num_l - 1];
   }
   printf("\n");
   
   n = hierarchy->fields[num_l - 1][0]->n;
   V_n    = hierarchy->fields[num_l - 1][2]->f1d;
   V_np1  = hierarchy->fields[num_l - 1][5]->f1d;
   for (i = 0; i < n; i++) {
      V_n[i] = V_np1[i]; 
   }

   double dt = 0.02 * dx;
   double time = 0.5;
   double time2 = 0;
   int max = 50;
   int itt;
   
   for (j = 0; j < time/dt; j++)
   {
      resid = 1;
      itt = 0;
      while(resid > 1e-8 || (resid > 1e-5 && itt > max-10)) {
         hierarchy->vCycle(4, dt, f_schrodinger_newton_opp_noPsi_1d, 
                 f_schrodinger_newton_jac_noPsi_1d);
         hierarchy->vCycle(4, dt, f_schrodinger_newton_opp_noV_1d, 
                 f_schrodinger_newton_jac_noV_1d);
         printf("change %e, residual %e\n", hierarchy->change[num_l - 1], 
                 hierarchy->resid[num_l - 1]);
         resid = hierarchy->change[num_l - 1];
         itt++;
         if (itt > max){
            printf("failed at t=%lf\n",time2);
            j=time/dt;
            break;
         }
      }
      time2+=dt;
      printf("\n");
      
      hierarchy->switchFields(0,3);
      hierarchy->switchFields(1,4);
      hierarchy->switchFields(2,5);
   }
   
   hierarchy->switchFields(0,3);
   hierarchy->switchFields(1,4);
   hierarchy->switchFields(2,5);
   
   double sum = 0;
   for (i = 0; i < n; i++)
   {
      s1_np1 = hierarchy->fields[num_l - 1][0]->f1d;
      s2_np1 = hierarchy->fields[num_l - 1][1]->f1d;
      V_np1 = hierarchy->fields[num_l - 1][2]->f1d;
      x = hierarchy->coords[num_l - 1]->mesh[0]->f1d;
      sum += (s1_np1[i]*s1_np1[i] + s2_np1[i]*s2_np1[i])*x[i]*x[i]*dx;
      printf("%10.7e\t%10.7e\n", (s1_np1[i]*s1_np1[i] + s2_np1[i]*s2_np1[i]), V_np1[i]);
   }
   printf("sum:%lf\n", sum);
   printf("\n");
   
   printf("after:\n");
#ifdef OMP_PARALLEL
   end = omp_get_wtime();
#else
   end_clock = clock();
   end = (double) end_clock/CLOCKS_PER_SEC;
#endif
   time_spent = (end - begin);
   printf("time spent: %lf\n", time_spent);

   //hierarchy->print(num_l-1);
   f_schrodinger_newton_indep_resid_1d(dt, hierarchy->coords[num_l-1], hierarchy->fields[num_l-1]);
   
   delete hierarchy;

   return 0;
}
