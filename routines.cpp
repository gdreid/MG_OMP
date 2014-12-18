#include <cmath>
#include <stdio.h>
#include <omp.h>
#include "Mesh.hpp"
#include "Hierarchy.hpp"
#include "Coords.hpp"
#include "routines.hpp"

/*
 * 
 * 
 * 
 * 1D Non Linear Laplace Equation
 * 
 * 
 * 
 */

/*
 * Operator for computing residuals for the 1D linear (phi1) and
 * non linear (phi2) Poisson equation with Dirichlet boundary 
 * conditions. Uses second order central differencing.
 */
void f_poisson_opp_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx, num_f;
   double f, dx;
  
   double* x = coords->mesh[0]->f1d;

   num_f = 2;
   double* phi1 = level[0+Hierarchy::u_i*num_f]->f1d;
   double* phi2 = level[1+Hierarchy::u_i*num_f]->f1d;

   double* phi1_res = level[0+Hierarchy::res_i*num_f]->f1d;
   double* phi2_res = level[1+Hierarchy::res_i*num_f]->f1d;

   dx = x[1] - x[0];
   nx = coords->size[0];

#   pragma omp parallel firstprivate(i, nx, f, dx, x, phi1, phi1_res, \
      phi2, phi2_res)
   {
      double phi1_xx, phi2_xx;
      
#      pragma  omp for nowait
      for (i = 1; i < nx - 1; i++) {
         f = phi1[i] * phi1[i];
         phi1_xx = (phi1[i - 1] - 2.0 * phi1[i] + phi1[i + 1]) / (dx * dx);
         phi1_res[i] = phi1_xx - f;

         f = 0.0;
         phi2_xx = (phi2[i - 1] - 2.0 * phi2[i] + phi2[i + 1]) / (dx * dx);
         phi2_res[i] = phi2_xx - f;
      }
   }

   phi1_res[0] = phi1[0] - 1.0;
   phi1_res[nx - 1] = phi1[nx - 1] - 0.0;

   phi2_res[0] = phi2[0] - 1.0;
   phi2_res[nx - 1] = phi2[nx - 1] - 1.0;
}

/*
 * Jacobian for the 1D Poisson equation 
 */
void f_poisson_jac_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx, num_f;
   double f, dx;

   double* x = coords->mesh[0]->f1d;

   num_f = 2;
   double* phi1 = level[0+Hierarchy::u_i*num_f]->f1d;
   double* phi2 = level[1+Hierarchy::u_i*num_f]->f1d;

   double* phi1_jac = level[0+Hierarchy::jac_i*num_f]->f1d;
   double* phi2_jac = level[1+Hierarchy::jac_i*num_f]->f1d;

   dx = x[1] - x[0];
   nx = coords->size[0];

#   pragma omp parallel firstprivate(i, nx, f, dx, x, phi1, phi2, phi1_jac, phi2_jac)
   {
      double dphi_phi1_xx, dphi_phi1;
      double dphi_phi2_xx, dphi_phi2;
      
#      pragma  omp for nowait
      for (i = 1; i < nx - 1; i++) {
         dphi_phi1_xx = (-2.0) / (dx * dx);
         dphi_phi1 = -2.0 * phi1[i];
         phi1_jac[i] = dphi_phi1_xx + dphi_phi1;

         dphi_phi2_xx = (-2.0) / (dx * dx);
         dphi_phi2 = 0.0;
         phi2_jac[i] = dphi_phi2_xx + dphi_phi2;
      }
   }

   phi1_jac[0] = 1.0;
   phi1_jac[nx - 1] = 1.0;

   phi2_jac[0] = 1.0;
   phi2_jac[nx - 1] = 1.0;
}

/*
 * Independent residual operator for the 1D Poisson equation. Uses fourth 
 * order central finite differencing. 
 */
void f_poisson_indep_resid_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx;
   double f, dx, res;
   double phi1_xx, phi2_xx;

   double* x = coords->mesh[0]->f1d;

   double* phi1 = level[0]->f1d;
   double* phi2 = level[1]->f1d;

   double* phi1_res = level[2]->f1d;
   double* phi2_res = level[3]->f1d;
   
   dx = x[1] - x[0];
   nx = coords->size[0];

   res = 0;
   for (i = 2; i < nx - 2; i++) {
      f = phi1[i] * phi1[i];
      phi1_xx = (-1.0/12.0 * phi1[i-2] + 4.0/3.0 * phi1[i-1] - 5.0/2.0*phi1[i] 
         + 4.0/3.0 * phi1[i+1] - 1.0/12.0 * phi1[i+2]) / (dx * dx);
      //phi1_xx = (phi1[i - 1] - 2.0 * phi1[i] + phi1[i + 1]) / (dx * dx);
      res += (phi1_xx - f)*(phi1_xx - f); 

      f = 0.0;
      phi2_xx = (-1.0/12.0 * phi2[i-2] + 4.0/3.0 * phi2[i-1] - 5.0/2.0*phi2[i] 
         + 4.0/3.0 * phi2[i+1] -1.0/12.0 * phi2[i+2]) / (dx * dx);
      //phi2_xx = (phi2[i - 1] - 2.0 * phi2[i] + phi2[i + 1]) / (dx * dx);
      res += (phi2_xx - f)*(phi2_xx - f); 
   }
   
   res /= ((double) (nx-4));
   res = sqrt(res);
   printf("independent residual evaluator %10.5e\n", res);
}

/*
 * 
 * 
 * 
 * 3D Non Linear Laplace Equation
 * 
 * 
 * 
 */

/*
 * Operator for computing residuals for the 3D non linear Poisson 
 * equation with Dirichlet boundary conditions. Uses second order 
 * central differencing.
 */
void f_poisson_opp_3d(double dt, Coords* coords, Mesh ** level) {
   int i, j, k, nx, ny, nz;
   double f, dx, dy, dz;

   double* x = coords->mesh[0]->f1d;
   double* y = coords->mesh[1]->f1d;
   double* z = coords->mesh[2]->f1d;

   double*** phi = level[0]->f3d;
   double*** phi_res = level[1]->f3d;

   dx = x[1] - x[0];
   dy = y[1] - y[0];
   dz = z[1] - z[0];
   nx = coords->size[0];
   ny = coords->size[1];
   nz = coords->size[2];

#   pragma omp parallel firstprivate(i, j, k, nx, ny, nz, dx, dy, dz, x, y, z phi, phi_res, f)
   {
      double phi_xx, phi_yy, phi_zz;
#      pragma  omp for nowait
      for (i = 1; i < nx - 1; i++) {
         for (j = 1; j < ny - 1; j++) {
            for (k = 1; k < nz - 1; k++) {

               f = phi[i][j][k] * phi[i][j][k];
               phi_xx = (phi[i - 1][j][k] - 2.0 * phi[i][j][k] 
                       + phi[i + 1][j][k]) / (dx * dx);
               phi_yy = (phi[i][j - 1][k] - 2.0 * phi[i][j][k] 
                       + phi[i][j + 1][k]) / (dy * dy);
               phi_zz = (phi[i][j][k - 1] - 2.0 * phi[i][j][k] 
                       + phi[i][j][k + 1]) / (dz * dz);
               phi_res[i][j][k] = phi_xx + phi_yy + phi_zz - f;
            }
         }
      }
   }
   
#   pragma omp parallel firstprivate(i, j, k, nx, ny, nz, dx, dy, dz, phi, phi_res, f)
   {
#      pragma  omp for nowait
      for (j = 0; j < ny; j++)
      {
         for (k = 0; k < nz; k++)
         {
            phi_res[0][j][k]    = phi[0][j][k] - 1.0;
            phi_res[nx-1][j][k] = phi[nx-1][j][k] - 1.0;
         }
      }
   }
   
#   pragma omp parallel firstprivate(i, j, k, nx, ny, nz, dx, dy, dz, phi, phi_res, f)
   {
#      pragma  omp for nowait
      for (i = 0; i < nx; i++)
      {
         for (k = 0; k < nz; k++)
         {
            phi_res[i][0][k]    = phi[i][0][k] - 1.0;
            phi_res[i][ny-1][k] = phi[i][ny-1][k] - 1.0;
         }
      }
   }
   
#   pragma omp parallel firstprivate(i, j, k, nx, ny, nz, dx, dy, dz, phi, phi_res, f)
   {
#      pragma  omp for nowait
      for (i = 0; i < nx; i++)
      {
         for (j = 0; j < ny; j++)
         {
            phi_res[i][j][0]    = phi[i][j][0] - 1.0;
            phi_res[i][j][nz-1] = phi[i][j][nz-1] - 1.0;
         }
      }
   }
}

/*
 * Jacobian for the 3D Poisson equation 
 */
void f_poisson_jac_3d(double dt, Coords* coords, Mesh ** level) {
   int i, j, k, nx, ny, nz;
   double f, dx, dy, dz;

   double* x = coords->mesh[0]->f1d;
   double* y = coords->mesh[1]->f1d;
   double* z = coords->mesh[2]->f1d;

   double*** phi = level[0]->f3d;
   double*** phi_jac = level[2]->f3d;

   dx = x[1] - x[0];
   dy = y[1] - y[0];
   dz = z[1] - z[0];
   nx = coords->size[0];
   ny = coords->size[1];
   nz = coords->size[2];

#   pragma omp parallel firstprivate(i, j, k, nx, ny, nz, dx, dy, dz, phi, phi_jac, f)
   {
      double phi_xx, phi_yy, phi_zz;
#      pragma  omp for nowait
      for (i = 1; i < nx - 1; i++) {
         for (j = 1; j < ny - 1; j++) {
            for (k = 1; k < nz - 1; k++) {

               f = 2*phi[i][j][k];
               phi_xx = - 2.0 / (dx * dx);
               phi_yy = - 2.0 / (dy * dy);
               phi_zz = - 2.0 / (dz * dz);
               phi_jac[i][j][k] = phi_xx + phi_yy + phi_zz - f;
            }
         }
      }
   }
   
#   pragma omp parallel firstprivate(i, j, k, nx, ny, nz, dx, dy, dz, phi, phi_jac, f)
   {
#      pragma  omp for nowait
      for (j = 0; j < ny; j++)
      {
         for (k = 0; k < nz; k++)
         {
            phi_jac[0][j][k]    = 1.0;
            phi_jac[nx-1][j][k] = 1.0;
         }
      }
   }
   
#   pragma omp parallel firstprivate(i, j, k, nx, ny, nz, dx, dy, dz, phi, phi_jac, f)
   {
#      pragma  omp for nowait
      for (i = 0; i < nx; i++)
      {
         for (k = 0; k < nz; k++)
         {
            phi_jac[i][0][k]    = 1.0;
            phi_jac[i][ny-1][k] = 1.0;
         }
      }
   }
   
#   pragma omp parallel firstprivate(i, j, k, nx, ny, nz, dx, dy, dz, phi, phi_jac, f)
   {
#      pragma  omp for nowait
      for (i = 0; i < nx; i++)
      {
         for (j = 0; j < ny; j++)
         {
            phi_jac[i][j][0]    = 1.0;
            phi_jac[i][j][nz-1] = 1.0;
         }
      }
   }
}

/*
 * Independent residual operator for the 3D Poisson equation. Uses second 
 * order central finite differencing to compute the derivatives and a second
 * order average to compute the source term 
 */
void f_poisson_indep_resid_3d(double dt, Coords* coords, Mesh ** level) {
   int i, j, k, nx, ny, nz;
   double f, dx, dy, dz;

   double* x = coords->mesh[0]->f1d;
   double* y = coords->mesh[1]->f1d;
   double* z = coords->mesh[2]->f1d;

   double*** phi = level[0]->f3d;
   double*** phi_res = level[1]->f3d;

   dx = x[1] - x[0];
   dy = y[1] - y[0];
   dz = z[1] - z[0];
   nx = coords->size[0];
   ny = coords->size[1];
   nz = coords->size[2];

   double phi_xx, phi_yy, phi_zz;
   double res, temp;
   res = 0;
   for (i = 2; i < nx - 2; i++) {
      for (j = 2; j < ny - 2; j++) {
         for (k = 2; k < nz - 2; k++) {
            
            f = phi[i][j][k] * phi[i][j][k];
            phi_xx = (-1.0/12.0 * phi[i-2][j][k] + 4.0/3.0 * phi[i-1][j][k] 
               - 5.0/2.0*phi[i][j][k] + 4.0/3.0 * phi[i+1][j][k] 
               - 1.0/12.0 * phi[i+2][j][k]) / (dx * dx);

            phi_yy = (-1.0/12.0 * phi[i][j-2][k] + 4.0/3.0 * phi[i][j-1][k] 
               - 5.0/2.0*phi[i][j][k] + 4.0/3.0 * phi[i][j+1][k] 
               - 1.0/12.0 * phi[i][j+2][k]) / (dy * dy);
            
            phi_zz = (-1.0/12.0 * phi[i][j][k-2] + 4.0/3.0 * phi[i][j][k-1] 
               - 5.0/2.0*phi[i][j][k] + 4.0/3.0 * phi[i][j][k+1] 
               - 1.0/12.0 * phi[i][j][k+2]) / (dz * dz);
            
            f = (phi[i-1][j][k] + phi[i+1][j][k] + phi[i][j-1][k] + phi[i][j+1][k] + phi[i][j][k-1] + phi[i][j][k+1])/6.0;
            f = f*f;
            phi_xx = (phi[i - 1][j][k] - 2.0 * phi[i][j][k] 
                       + phi[i + 1][j][k]) / (dx * dx);
            phi_yy = (phi[i][j - 1][k] - 2.0 * phi[i][j][k] 
                       + phi[i][j + 1][k]) / (dy * dy);
            phi_zz = (phi[i][j][k - 1] - 2.0 * phi[i][j][k] 
                       + phi[i][j][k + 1]) / (dz * dz);
            
            temp = phi_xx + phi_yy + phi_zz - f;
            res += temp*temp;
            if (i == nx/2 && j == ny/2 && k == nz/2)
            {
            printf("%d %d %d %lf\n", i, j, k, temp);
            }
         }
      }
   }
   res /= ((double) ((nx-4)*(ny-4)*(nz-4)));
   res = sqrt(res);
   printf("independent residual evaluator %10.5e\n", res);
}

/*
 * 
 * 
 * 
 * 1D Schrodinger Equation 
 * Equations have been diagonalized to solve non dominant problem
 * 
 * 
 * 
 */

/*
 * Operator for computing residuals for the 1D linear Schrodinger equation
 * on the line with Dirichlet boundary conditions. Uses second order, central
 * spatial differencing and Crank-Nicolson differencing in time. 
 * 
 * Since the most straightforward Crank-Nicolson discretization is not 
 * diagonally dominant, the equations have been algebraically rearranged.
 */
void f_schrodinger_opp_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx;
   double f, dx;

   double* x = coords->mesh[0]->f1d;

   double* sigma1_n = level[0]->f1d;
   double* sigma2_n = level[1]->f1d;
   double* sigma1_np1 = level[2]->f1d;
   double* sigma2_np1 = level[3]->f1d;

   double* sigma1_n_res = level[4]->f1d;
   double* sigma2_n_res = level[5]->f1d;
   double* sigma1_np1_res = level[6]->f1d;
   double* sigma2_np1_res = level[7]->f1d;

   dx = x[1] - x[0];
   nx = coords->size[0];

#   pragma omp parallel firstprivate(i, nx, dx, sigma1_n, sigma2_n, \
      sigma1_np1, sigma2_np1, sigma1_n_res, sigma2_n_res, sigma1_np1_res, \
      sigma2_np1_res)
   {           
#      pragma  omp for nowait
      for (i = 1; i < nx - 1; i++) {
         sigma1_np1_res[i] = (-sigma1_np1[i]+sigma1_n[i])*(dx*dx)/(dt*dt)+((1.0/2.0)
                 *sigma2_n[i-1]-2.0*sigma2_n[i]+(1.0/2.0)*sigma2_n[i+1]+(1.0/2.0)
                 *sigma2_np1[i-1]+(1.0/2.0)*sigma2_np1[i+1])/dt+((1.0/2.0)
                 *sigma1_n[i-1]-sigma1_n[i]+(1.0/2.0)*sigma1_n[i+1]+(1.0/2.0)
                 *sigma1_np1[i-1]-sigma1_np1[i]+(1.0/2.0)*sigma1_np1[i+1])/(dx*dx);
         sigma2_np1_res[i] = (sigma2_np1[i]-sigma2_n[i])*(dx*dx)/(dt*dt)+((1.0/2.0)
                 *sigma1_n[i-1]-2.0*sigma1_n[i]+(1.0/2.0)*sigma1_n[i+1]+(1.0/2.0)
                 *sigma1_np1[i-1]+(1.0/2.0)*sigma1_np1[i+1])/dt+(-(1.0/2.0)
                 *sigma2_np1[i-1]-(1.0/2.0)*sigma2_n[i-1]+sigma2_n[i]-(1.0/2.0)
                 *sigma2_n[i+1]+sigma2_np1[i]-(1.0/2.0)*sigma2_np1[i+1])/(dx*dx);
         sigma1_n_res[i] = 0;
         sigma2_n_res[i] = 0;
      }
   }

   sigma1_n_res[0] = sigma1_n[0] - 0.0;
   sigma1_n_res[nx - 1] = sigma1_n[nx - 1] - 0.0;

   sigma2_n_res[0] = sigma2_n[0] - 0.0;
   sigma2_n_res[nx - 1] = sigma2_n[nx - 1] - 0.0;
   
   sigma1_np1_res[0] = sigma1_np1[0] - 0.0;
   sigma1_np1_res[nx - 1] = sigma1_np1[nx - 1] - 0.0;

   sigma2_np1_res[0] = sigma2_np1[0] - 0.0;
   sigma2_np1_res[nx - 1] = sigma2_np1[nx - 1] - 0.0;
}

/*
 * Jacobian for the 1D Schrodinger equation 
 */
void f_schrodinger_jac_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx;
   double f, dx;

   double* x = coords->mesh[0]->f1d;

   double* sigma1_n = level[0]->f1d;
   double* sigma2_n = level[1]->f1d;
   double* sigma1_np1 = level[2]->f1d;
   double* sigma2_np1 = level[3]->f1d;

   double* sigma1_n_jac = level[8]->f1d;
   double* sigma2_n_jac = level[9]->f1d;
   double* sigma1_np1_jac = level[10]->f1d;
   double* sigma2_np1_jac = level[11]->f1d;

   dx = x[1] - x[0];
   nx = coords->size[0];

#   pragma omp parallel firstprivate(i, nx, dx, sigma1_n, sigma2_n, \
      sigma1_np1, sigma2_np1, sigma1_n_jac, sigma2_n_jac, sigma1_np1_jac, \
      sigma2_np1_jac)
   {     
#      pragma  omp for nowait
      for (i = 1; i < nx - 1; i++) {
         
         sigma1_np1_jac[i] = -(dx*dx)/(dt*dt)-1.0/(dx*dx);
         sigma2_np1_jac[i] = (dx*dx)/(dt*dt)+1.0/(dx*dx);
         sigma1_n_jac[i] = 1;
         sigma2_n_jac[i] = 1;
      }
   }

   sigma1_n_jac[0] = 1.0;
   sigma1_n_jac[nx - 1] = 1.0;

   sigma2_n_jac[0] = 1.0;
   sigma2_n_jac[nx - 1] = 1.0;
   
   sigma1_np1_jac[0] = 1.0;
   sigma1_np1_jac[nx - 1] = 1.0;

   sigma2_np1_jac[0] = 1.0;
   sigma2_np1_jac[nx - 1] = 1.0;
}

/*
 * Independent residual operator for the 1D Schrodinger equation on a line. 
 * Uses fourth order finite differencing for the second derivatives and the
 * straightforward, non diagonalized Crank-Nicolson scheme for the residuals.
 */
void f_schrodinger_indep_resid_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx;
   double f, dx;

   double* x = coords->mesh[0]->f1d;

   double* sigma1_n = level[0]->f1d;
   double* sigma2_n = level[1]->f1d;
   double* sigma1_np1 = level[2]->f1d;
   double* sigma2_np1 = level[3]->f1d;

   dx = x[1] - x[0];
   nx = coords->size[0];

   double sigma1_n_xx, sigma1_np1_xx, sigma2_n_xx, sigma2_np1_xx;
   double sigma1_xx, sigma2_xx;
   double V_sigma1_n, V_sigma2_n, V_sigma1_np1, V_sigma2_np1;
   double V_sigma1, V_sigma2;
   double dt_sigma1, dt_sigma2;
   double res, temp1, temp2;

   res = 0;
   for (i = 2; i < nx - 2; i++) {

      sigma1_n_xx = (-1.0/12.0 * sigma1_n[i-2] + 4.0/3.0 * sigma1_n[i-1] - 5.0/2.0*sigma1_n[i] 
         + 4.0/3.0 * sigma1_n[i+1] - 1.0/12.0 * sigma1_n[i+2]) / (dx * dx);
      
      sigma2_n_xx = (-1.0/12.0 * sigma2_n[i-2] + 4.0/3.0 * sigma2_n[i-1] - 5.0/2.0*sigma2_n[i] 
         + 4.0/3.0 * sigma2_n[i+1] - 1.0/12.0 * sigma2_n[i+2]) / (dx * dx);
      
      sigma1_np1_xx = (-1.0/12.0 * sigma1_np1[i-2] + 4.0/3.0 * sigma1_np1[i-1] - 5.0/2.0*sigma1_np1[i] 
         + 4.0/3.0 * sigma1_np1[i+1] - 1.0/12.0 * sigma1_np1[i+2]) / (dx * dx);
      
      sigma2_np1_xx = (-1.0/12.0 * sigma2_np1[i-2] + 4.0/3.0 * sigma2_np1[i-1] - 5.0/2.0*sigma2_np1[i] 
         + 4.0/3.0 * sigma2_np1[i+1] - 1.0/12.0 * sigma2_np1[i+2]) / (dx * dx);

      sigma1_xx = 0.5 * (sigma1_n_xx + sigma1_np1_xx);
      sigma2_xx = 0.5 * (sigma2_n_xx + sigma2_np1_xx);

      V_sigma1_n = 0 * sigma1_n[i];
      V_sigma1_np1 = 0 * sigma1_np1[i];
      V_sigma1 = 0.5 * (V_sigma1_n + V_sigma1_np1);

      V_sigma2_n = 0 * sigma2_n[i];
      V_sigma2_np1 = 0 * sigma2_np1[i];
      V_sigma2 = 0.5 * (V_sigma2_n + V_sigma2_np1);

      dt_sigma1 = (sigma1_np1[i] - sigma1_n[i]) / dt;
      dt_sigma2 = (sigma2_np1[i] - sigma2_n[i]) / dt;

      temp1 = dt_sigma2 + sigma1_xx - V_sigma1;
      temp2 = dt_sigma1 - sigma2_xx + V_sigma2;
      res += temp1*temp1+temp2*temp2;
   }
   res /= ((double) (nx-2));
   res = sqrt(res);
   printf("independent residual evaluator %10.5e\n", res);
}

/*
 * 
 * 
 * 
 * 1D Schrodinger-Newton Equation (in r) 
 * Equations have been diagonalized to solve non dominant problem
 * 
 * 
 * 
 */

/*
 * Operator for computing residuals for the spherically symmetric Schrodinger-
 * Newton equations with Dirichlet boundary conditions. Uses second order, 
 * central spatial differencing and Crank-Nicolson differencing in time. 
 * 
 * Only updates the Schrodinger Field residuals
 * 
 * Since the most straightforward Crank-Nicolson discretization is not 
 * diagonally dominant, the equations have been algebraically rearranged.
 * 
 * The time update has been separated from the computation of the potential
 * residuals for the same reason; for large time steps the equations stop being 
 * diagonally dominant. By separating the update into an update of the 
 * Schrodinger fields followed by n update of the potential fields, better 
 * stability is achieved.
 */
void f_schrodinger_newton_opp_noV_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx;
   double r, dr;

   double* x = coords->mesh[0]->f1d;

   double* s1_n = level[0]->f1d;
   double* s2_n = level[1]->f1d;
   double* V_n  = level[2] ->f1d;
   double* s1_np1 = level[3]->f1d;
   double* s2_np1 = level[4]->f1d;
   double* V_np1  = level[5]->f1d;

   double* s1_n_res = level[6]->f1d;
   double* s2_n_res = level[7]->f1d;
   double* V_n_res  = level[8]->f1d;
   double* s1_np1_res = level[9]->f1d;
   double* s2_np1_res = level[10]->f1d;
   double* V_np1_res  = level[11]->f1d;

   dr = x[1] - x[0];
   nx = coords->size[0];

   if (nx > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(i, nx, dr, dt, x, s1_n, s2_n, V_n, \
      s1_np1, s2_np1, V_np1, s1_n_res, s2_n_res, V_n_res, s1_np1_res, \
      s2_np1_res, V_np1_res)
      {
#         pragma  omp for nowait
         for (i = 1; i < nx - 1; i++) {
            double r = x[i];

            s1_np1_res[i] = -(2 * ((1.0 / 4.0)*(s1_np1[i + 1] - s1_np1[i - 1]) 
               / dr + (1.0 / 4.0)*(s1_n[i + 1] - s1_n[i - 1]) / dr)) / r 
               - (1.0 / 2.0)*(s1_np1[i + 1] - 2 * s1_np1[i] + s1_np1[i - 1]) 
               / (dr * dr)-(1.0 / 2.0)*(s1_n[i + 1] - 2 * s1_n[i] + s1_n[i - 1]) 
               / (dr * dr)+(1.0 / 2.0) * V_np1[i] * s1_np1[i]+(1.0 / 2.0) * V_n[i] 
               * s1_n[i]+(-(-dr * s2_np1[i + 1] + dr * s2_np1[i - 1] - dr 
               * s2_n[i + 1] + dr * s2_n[i - 1] - r * s2_np1[i + 1] - r 
               * s2_np1[i - 1] - r * s2_n[i + 1] + 2 * r * s2_n[i] - r 
               * s2_n[i - 1] + V_n[i] * s2_n[i] * r * (dr * dr)) / (r * (2 
               + V_np1[i]*(dr * dr))) - s2_n[i]) / dt - (-2 * r * (dr * dr) 
               * s1_np1[i] + 2 * r * (dr * dr) * s1_n[i]) / (r * (2 + V_np1[i]
               *(dr * dr))*(dt * dt));
            
            s2_np1_res[i] = (2 * ((1.0 / 4.0)*(s2_np1[i + 1] - s2_np1[i - 1]) 
               / dr + (1.0 / 4.0)*(s2_n[i + 1] - s2_n[i - 1]) / dr)) / r + (1.0 
               / 2.0)*(s2_np1[i + 1] - 2 * s2_np1[i] + s2_np1[i - 1]) / (dr 
               * dr)+(1.0 / 2.0)*(s2_n[i + 1] - 2 * s2_n[i] + s2_n[i - 1]) 
               / (dr * dr)-(1.0 / 2.0) * V_np1[i] * s2_np1[i]-(1.0 / 2.0) 
               * V_n[i] * s2_n[i]+(-(-dr * s1_np1[i + 1] + dr * s1_np1[i - 1] 
               - dr * s1_n[i + 1] + dr * s1_n[i - 1] - r * s1_np1[i + 1] - r 
               * s1_np1[i - 1] - r * s1_n[i + 1] + 2 * r * s1_n[i] - r 
               * s1_n[i - 1] + V_n[i] * s1_n[i] * r * (dr * dr)) / (r * (2 
               + V_np1[i]*(dr * dr))) - s1_n[i]) / dt - (2 * r * (dr * dr) 
               * s2_np1[i] - 2 * r * (dr * dr) * s2_n[i]) / (r * (2 + V_np1[i]
               *(dr * dr))*(dt * dt));
            
            V_np1_res[i] = 0.0;

            s1_n_res[i] = 0;
            s2_n_res[i] = 0;
            V_n_res[i] = 0;
         }
      }
   } else {
      for (i = 1; i < nx - 1; i++) {
         double r = x[i];

         s1_np1_res[i] = -(2 * ((1.0 / 4.0)*(s1_np1[i + 1] - s1_np1[i - 1]) 
            / dr + (1.0 / 4.0)*(s1_n[i + 1] - s1_n[i - 1]) / dr)) / r - (1.0 
            / 2.0)*(s1_np1[i + 1] - 2 * s1_np1[i] + s1_np1[i - 1]) / (dr * dr)
            -(1.0 / 2.0)*(s1_n[i + 1] - 2 * s1_n[i] + s1_n[i - 1]) / (dr * dr)
            +(1.0 / 2.0) * V_np1[i] * s1_np1[i]+(1.0 / 2.0) * V_n[i] * s1_n[i]
            +(-(-dr * s2_np1[i + 1] + dr * s2_np1[i - 1] - dr * s2_n[i + 1] 
            + dr * s2_n[i - 1] - r * s2_np1[i + 1] - r * s2_np1[i - 1] - r 
            * s2_n[i + 1] + 2 * r * s2_n[i] - r * s2_n[i - 1] + V_n[i] * s2_n[i] 
            * r * (dr * dr)) / (r * (2 + V_np1[i]*(dr * dr))) - s2_n[i]) / dt 
            - (-2 * r * (dr * dr) * s1_np1[i] + 2 * r * (dr * dr) * s1_n[i]) 
            / (r * (2 + V_np1[i]*(dr * dr))*(dt * dt));
         
         s2_np1_res[i] = (2 * ((1.0 / 4.0)*(s2_np1[i + 1] - s2_np1[i - 1]) / dr 
            + (1.0 / 4.0)*(s2_n[i + 1] - s2_n[i - 1]) / dr)) / r + (1.0 / 2.0)
            *(s2_np1[i + 1] - 2 * s2_np1[i] + s2_np1[i - 1]) / (dr * dr)+(1.0 
            / 2.0)*(s2_n[i + 1] - 2 * s2_n[i] + s2_n[i - 1]) / (dr * dr)-(1.0 
            / 2.0) * V_np1[i] * s2_np1[i]-(1.0 / 2.0) * V_n[i] * s2_n[i]+(-(-dr 
            * s1_np1[i + 1] + dr * s1_np1[i - 1] - dr * s1_n[i + 1] + dr 
            * s1_n[i - 1] - r * s1_np1[i + 1] - r * s1_np1[i - 1] - r 
            * s1_n[i + 1] + 2 * r * s1_n[i] - r * s1_n[i - 1] + V_n[i] * s1_n[i] 
            * r * (dr * dr)) / (r * (2 + V_np1[i]*(dr * dr))) - s1_n[i]) / dt 
            - (2 * r * (dr * dr) * s2_np1[i] - 2 * r * (dr * dr) * s2_n[i]) 
            / (r * (2 + V_np1[i]*(dr * dr))*(dt * dt));
         
         V_np1_res[i] = 0.0;

         s1_n_res[i] = 0;
         s2_n_res[i] = 0;
         V_n_res[i] = 0;
      }
   }

   s1_n_res[0] = 0.0;
   s1_n_res[nx - 1] = 0.0;

   s2_n_res[0] = 0.0;
   s2_n_res[nx - 1] = 0.0;
   
   V_n_res[0] = 0.0;
   V_n_res[nx - 1] = 0.0;
   
   s1_np1_res[0] = -3.0 * s1_np1[0] + 4.0 * s1_np1[1] - 1.0 * s1_np1[2];
   s1_np1_res[nx - 1] = s1_np1[nx - 1] - 0.0;

   s2_np1_res[0] = -3.0 * s2_np1[0] + 4.0 * s2_np1[1] - 1.0 * s2_np1[2];
   s2_np1_res[nx - 1] = s2_np1[nx - 1] - 0.0;
   
   V_np1_res[0] = 0.0;
   V_np1_res[nx - 1] = 0.0;
}

/*
 * Jacobian for the Schrodinger fields of the spherically symmetric Schrodinger-
 * Newton equation. 
 */
void f_schrodinger_newton_jac_noV_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx;
   double r, dr;

   double* x = coords->mesh[0]->f1d;

   double* s1_n = level[0]->f1d;
   double* s2_n = level[1]->f1d;
   double* V_n  = level[2] ->f1d;
   double* s1_np1 = level[3]->f1d;
   double* s2_np1 = level[4]->f1d;
   double* V_np1  = level[5]->f1d;

   double* s1_n_jac = level[12]->f1d;
   double* s2_n_jac = level[13]->f1d;
   double* V_n_jac  = level[14]->f1d;
   double* s1_np1_jac = level[15]->f1d;
   double* s2_np1_jac = level[16]->f1d;
   double* V_np1_jac  = level[17]->f1d;

   dr = x[1] - x[0];
   nx = coords->size[0];

   if (nx > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(i, nx, dr, dt, x, s1_n, s2_n, V_n, \
      s1_np1, s2_np1, V_np1, s1_n_jac, s2_n_jac, V_n_jac, s1_np1_jac, \
      s2_np1_jac, V_np1_jac)
      {
#         pragma  omp for nowait
         for (i = 1; i < nx - 1; i++) {
            r = x[i];

            s1_np1_jac[i] = 1.0 / (dr * dr)+(1.0 / 2.0) * V_np1[i] + 2 * (dr 
               * dr) / ((2 + V_np1[i]*(dr * dr))*(dt * dt));
            s2_np1_jac[i] = -1.0 / (dr * dr)-(1.0 / 2.0) * V_np1[i] - 2 * (dr 
               * dr) / ((2 + V_np1[i]*(dr * dr))*(dt * dt));
            V_np1_jac[i] = 1.0;

            s1_n_jac[i] = 1.0;
            s2_n_jac[i] = 1.0;
            V_n_jac[i] = 1.0;
         }
      }
   } else {
      for (i = 1; i < nx - 1; i++) {
         r = x[i];

         s1_np1_jac[i] = 1.0 / (dr * dr)+(1.0 / 2.0) * V_np1[i] + 2 * (dr 
            * dr) / ((2 + V_np1[i]*(dr * dr))*(dt * dt));
         s2_np1_jac[i] = -1.0 / (dr * dr)-(1.0 / 2.0) * V_np1[i] - 2 * (dr 
            * dr) / ((2 + V_np1[i]*(dr * dr))*(dt * dt));
         V_np1_jac[i] = 1.0;

         s1_n_jac[i] = 1.0;
         s2_n_jac[i] = 1.0;
         V_n_jac[i] = 1.0;
      }
   }

   s1_n_jac[0] = 1.0;
   s1_n_jac[nx - 1] = 1.0;

   s2_n_jac[0] = 1.0;
   s2_n_jac[nx - 1] = 1.0;
   
   V_n_jac[0] = 1.0;
   V_n_jac[nx - 1] = 1.0;
   
   s1_np1_jac[0] = -3.0;
   s1_np1_jac[nx - 1] = 1.0;

   s2_np1_jac[0] = -3.0;
   s2_np1_jac[nx - 1] = 1.0;
   
   V_np1_jac[0] = 1.0;
   V_np1_jac[nx - 1] = 1.0;
}

/*
 * Independent residual operator for the spherically symmetric Schrodinger-
 * Newton equation equation. Uses second order finite differencing for the 
 * second derivatives and the straightforward, non diagonalized Crank-Nicolson 
 * scheme for the residuals.
 */
void f_schrodinger_newton_indep_resid_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx;
   double r, dr;

   double* x = coords->mesh[0]->f1d;

   double* s1_n = level[0]->f1d;
   double* s2_n = level[1]->f1d;
   double* V_n  = level[2] ->f1d;
   double* s1_np1 = level[3]->f1d;
   double* s2_np1 = level[4]->f1d;
   double* V_np1  = level[5]->f1d;

   double* s1_n_res = level[6]->f1d;
   double* s2_n_res = level[7]->f1d;
   double* V_n_res  = level[8]->f1d;
   double* s1_np1_res = level[9]->f1d;
   double* s2_np1_res = level[10]->f1d;
   double* V_np1_res = level[11]->f1d;

   dr = x[1] - x[0];
   nx = coords->size[0];

   double temp1, temp2, temp3, res;
   res = 0;
   for (i = 2; i < nx - 2; i++) {
      double r = x[i];

      temp1 = (s2_np1[i]-s2_n[i])/dt-(2*((1.0/4.0)*(s1_np1[i+1]-s1_np1[i-1])/dr
         +(1.0/4.0)*(s1_n[i+1]-s1_n[i-1])/dr))/r-(1.0/2.0)*(-(1.0/12.0)
         *s1_np1[i+2]-(1.0/12.0)*s1_np1[i-2]+(4.0/3.0)*s1_np1[i+1]+(4.0/3.0)
         *s1_np1[i-1]-(5.0/2.0)*s1_np1[i])/(dr*dr)-(1.0/2.0)*(-(1.0/12.0)
         *s1_n[i+2]-(1.0/12.0)*s1_n[i-2]+(4.0/3.0)*s1_n[i+1]+(4.0/3.0)
         *s1_n[i-1]-(5.0/2.0)*s1_n[i])/(dr*dr)+(1.0/2.0)*V_np1[i]*s1_np1[i]
         +(1.0/2.0)*V_n[i]*s1_n[i];

      temp2 = (s1_np1[i]-s1_n[i])/dt+(2*((1.0/4.0)*(s2_np1[i+1]-s2_np1[i-1])/dr
         +(1.0/4.0)*(s2_n[i+1]-s2_n[i-1])/dr))/r+(1.0/2.0)*(-(1.0/12.0)
         *s2_np1[i+2]-(1.0/12.0)*s2_np1[i-2]+(4.0/3.0)*s2_np1[i+1]+(4.0/3.0)
         *s2_np1[i-1]-(5.0/2.0)*s2_np1[i])/(dr*dr)+(1.0/2.0)*(-(1.0/12.0)
         *s2_n[i+2]-(1.0/12.0)*s2_n[i-2]+(4.0/3.0)*s2_n[i+1]+(4.0/3.0)
         *s2_n[i-1]-(5.0/2.0)*s2_n[i])/(dr*dr)-(1.0/2.0)*V_np1[i]*s2_np1[i]
         -(1.0/2.0)*V_n[i]*s2_n[i]; 

      temp3 = (V_np1[i+1]-V_np1[i-1])/(r*dr)+(-(1.0/12.0)*V_np1[i+2]-(1.0/12.0)
         *V_np1[i-2]+(4.0/3.0)*V_np1[i+1]+(4.0/3.0)*V_np1[i-1]-(5.0/2.0)
         *V_np1[i])/(dr*dr)-(s1_np1[i]*s1_np1[i])-(s2_np1[i]*s2_np1[i]);
      
      res += temp1*temp1 + temp2*temp2 + temp3*temp3;
   }
   res /= ((double) (3*(nx-2)));
   res = sqrt(res);
   printf("independent residual evaluator %10.5e\n", res);
}

/*
 * Operator for computing residuals for the spherically symmetric Schrodinger-
 * Newton equations with Dirichlet boundary conditions. Uses second order, 
 * central spatial differencing on the advance time step. 
 * 
 * Only updates the Potential Field
 * 
 * The time update has been separated from the computation of the potential
 * residuals due to the off diagonal nature of the Jacobian; for large time 
 * steps the equations stop being diagonally dominant. By separating the update 
 * into an update of the Schrodinger fields followed by n update of the 
 * potential fields, better stability is achieved.
 */
void f_schrodinger_newton_opp_noPsi_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx;
   double r, dr;

   double* x = coords->mesh[0]->f1d;

   double* s1_n = level[0]->f1d;
   double* s2_n = level[1]->f1d;
   double* V_n  = level[2] ->f1d;
   double* s1_np1 = level[3]->f1d;
   double* s2_np1 = level[4]->f1d;
   double* V_np1  = level[5]->f1d;

   double* s1_n_res = level[6]->f1d;
   double* s2_n_res = level[7]->f1d;
   double* V_n_res  = level[8]->f1d;
   double* s1_np1_res = level[9]->f1d;
   double* s2_np1_res = level[10]->f1d;
   double* V_np1_res  = level[11]->f1d;
   
   dr = x[1] - x[0];
   nx = coords->size[0];

   if (nx > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(i, nx, dr, dt, x, s1_n, s2_n, V_n, \
      s1_np1, s2_np1, V_np1, s1_n_res, s2_n_res, V_n_res, s1_np1_res, \
      s2_np1_res, V_np1_res)
      {
#         pragma  omp for nowait
         for (i = 1; i < nx - 1; i++) {
            double r = x[i];

            s1_np1_res[i] = 0;
            s2_np1_res[i] = 0;
            V_np1_res[i] = (V_np1[i + 1] - V_np1[i - 1]) / (r * dr)
               +(V_np1[i + 1] - 2 * V_np1[i] + V_np1[i - 1]) / (dr * dr)
               -(s1_np1[i] * s1_np1[i])-(s2_np1[i] * s2_np1[i]);

            s1_n_res[i] = 0;
            s2_n_res[i] = 0;
            V_n_res[i] = 0;
         }
      }
   } else {
      for (i = 1; i < nx - 1; i++) {
         double r = x[i];

         s1_np1_res[i] = 0;
         s2_np1_res[i] = 0;
         V_np1_res[i] = (V_np1[i + 1] - V_np1[i - 1]) / (r * dr)+(V_np1[i + 1] 
            - 2 * V_np1[i] + V_np1[i - 1]) / (dr * dr)-(s1_np1[i] * s1_np1[i])
            -(s2_np1[i] * s2_np1[i]);

         s1_n_res[i] = 0;
         s2_n_res[i] = 0;
         V_n_res[i] = 0;
      }
   }

   s1_n_res[0] = 0.0;
   s1_n_res[nx - 1] = 0.0;

   s2_n_res[0] = 0.0;
   s2_n_res[nx - 1] = 0.0;
   
   V_n_res[0] = 0.0;
   V_n_res[nx - 1] = 0.0;
   
   s1_np1_res[0] = 0.0;;
   s1_np1_res[nx - 1] = 0.0;

   s2_np1_res[0] = 0.0;
   s2_np1_res[nx - 1] = 0.0;
   
   V_np1_res[0] = -3.0 * V_np1[0] + 4.0 * V_np1[1] - 1.0 * V_np1[2];
   V_np1_res[nx - 1] = V_np1[nx-1] - 0.0;
}

/*
 * Jacobian for the Potential field of the spherically symmetric Schrodinger-
 * Newton equation. 
 */
void f_schrodinger_newton_jac_noPsi_1d(double dt, Coords* coords, Mesh ** level) {
   int i, nx;
   double r, dr;

   double* x = coords->mesh[0]->f1d;

   double* s1_n = level[0]->f1d;
   double* s2_n = level[1]->f1d;
   double* V_n  = level[2] ->f1d;
   double* s1_np1 = level[3]->f1d;
   double* s2_np1 = level[4]->f1d;
   double* V_np1  = level[5]->f1d;

   double* s1_n_jac = level[12]->f1d;
   double* s2_n_jac = level[13]->f1d;
   double* V_n_jac  = level[14]->f1d;
   double* s1_np1_jac = level[15]->f1d;
   double* s2_np1_jac = level[16]->f1d;
   double* V_np1_jac  = level[17]->f1d;

   dr = x[1] - x[0];
   nx = coords->size[0];

   if (nx > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(i, nx, dr, dt, x, s1_n, s2_n, V_n, \
      s1_np1, s2_np1, V_np1, s1_n_jac, s2_n_jac, V_n_jac, s1_np1_jac, \
      s2_np1_jac, V_np1_jac)
      {
#         pragma  omp for nowait
         for (i = 1; i < nx - 1; i++) {
            r = x[i];

            s1_np1_jac[i] = 1.0;
            s2_np1_jac[i] = 1.0;
            V_np1_jac[i]  = -2.0 / (dr * dr);

            s1_n_jac[i] = 1.0;
            s2_n_jac[i] = 1.0;
            V_n_jac[i]  = 1.0;
         }
      }
   } else {
      for (i = 1; i < nx - 1; i++) {
         r = x[i];

         s1_np1_jac[i] = 1.0;
         s2_np1_jac[i] = 1.0;
         V_np1_jac[i]  = -2.0 / (dr * dr);

         s1_n_jac[i] = 1.0;
         s2_n_jac[i] = 1.0;
         V_n_jac[i]  = 1.0;
      }
   }

   s1_n_jac[0] = 1.0;
   s1_n_jac[nx - 1] = 1.0;

   s2_n_jac[0] = 1.0;
   s2_n_jac[nx - 1] = 1.0;
   
   V_n_jac[0] = 1.0;
   V_n_jac[nx - 1] = 1.0;
   
   s1_np1_jac[0] = 1.0;
   s1_np1_jac[nx - 1] = 1.0;

   s2_np1_jac[0] = 1.0;
   s2_np1_jac[nx - 1] = 1.0;
   
   V_np1_jac[0] = -3.0;
   V_np1_jac[nx - 1] = 1.0;
}