#include <stdio.h>
#include <cmath>
#include "Hierarchy.hpp"
#include "Mesh.hpp"
#include "Coords.hpp"

/*
 * Hierarchy is the basic multigrid structure consisting of num_l_ levels.
 * Contains routines o perform the vCycle and compute residuals.
 * 
 * num_l_:     number of levels
 * num_f_:     number of PDE fields (allocates 5x this amount for work)
 * dim_:       dimension of the problem (1,2,3)
 * size_base_: array containing the dimensions of the coarsest field. For a
 *             2D Hierarchy this would typically be size_base_={3,3,3}
 * bbox_:      array of length 2*dim_ containing the ordered boundaries of the 
 *             region. For example, a 2D grid defined on 0<x<1, 1<y<10 
 *             bbox_={0,1,1,10}
 */
Hierarchy::Hierarchy(int num_l_, int num_f_, int dim_, int *size_base_, double * bbox_) {
   int i, j;
   int size[3];

   num_l = num_l_;
   num_f = num_f_;
   num_w = 5 * num_f;
   dim = dim_;
   change = new double[num_l];
   resid = new double[num_l];
   for (i = 0; i < dim; i++) {
      size_base[i] = size_base_[i];
      size[i] = size_base[i];

      bbox[2 * i] = bbox_[2 * i];
      bbox[2 * i + 1] = bbox_[2 * i + 1];
   }

   fields = new Mesh**[num_l];
   coords = new Coords*[num_l];

   for (i = 0; i < num_l; i++) {
      fields[i] = new Mesh*[num_w];
      coords[i] = new Coords(dim, size, bbox);

      for (j = 0; j < num_w; j++) {
         fields[i][j] = new Mesh(dim, size);
      }

      for (j = 0; j < dim; j++) {
         size[j] = size[j] * 2 - 1;
      }
   }


}

Hierarchy::~Hierarchy() {
   int i, j;
   delete change;
   delete resid;
   for (i = 0; i < num_l; i++) {
      for (j = 0; j < num_w; j++) {
         delete fields[i][j];
      }
      delete [] fields[i];
      delete coords[i];
   }
   delete [] fields;
   delete [] coords;
}

/*
 * Performs restriction edge (fine to coarse) of the vCycle algorithm.
 * 
 * itt:    number of smoothing cycles to apply
 * dt:     time step
 * opp:    residual operator
 * jac:    jacobian operator
 */
void Hierarchy::restrictCycle(int itt, double dt,
        void (*opp)(double dt, Coords* coords, Mesh ** level),
        void (*jac)(double dt, Coords* coords, Mesh ** level)) {
   int lev, i;
   int field_u, field_s, field_res, field_jac;
   Mesh* c;
   Mesh* f;
   Mesh** level_c;
   Mesh** level_f;
   Coords* coords_c;
   Coords* coords_f;

   for (lev = num_l - 1; lev > 0; lev--) {
      relaxLevel(itt, dt, lev, opp, jac);
      level_c = fields[lev - 1];
      coords_c = coords[lev - 1];
      level_f = fields[lev];
      coords_f = coords[lev];

      //populate residual fields with L u
      opp(dt, coords_f, level_f);

      for (i = 0; i < num_f; i++) {
         field_u = i + u_i * num_f;
         field_s = i + s_i * num_f;
         field_res = i + res_i * num_f;
         field_jac = i + jac_i * num_f;

         //restrict residual fields to next level
         c = level_c[field_jac];
         f = level_f[field_res];
         Mesh::restrict(c, f);

         //restrict fields to next level
         c = level_c[field_u];
         f = level_f[field_u];
         Mesh::restrict(c, f);

         //restrict source to next level
         c = level_c[field_s];
         f = level_f[field_s];
         Mesh::restrict(c, f);
      }

      opp(dt, coords_c, level_c);
      calcSource(lev - 1);
   }
}

/*
 * Zeros the source fields 
 */
void Hierarchy::zeroSource(int lev) {
   int i, j, n;
   int field_s;
   Mesh** level_l;

   level_l = fields[lev];
   for (i = 0; i < num_f; i++) {
      field_s = i + s_i * num_f;
      n = level_l[0]->n;

      if (n > MIN_GRID_SIZE) {
#         pragma omp parallel firstprivate(level_l, field_s, n, j)
         {
#            pragma  omp for nowait
            for (j = 0; j < n; j++) {
               level_l[field_s]->data[j] = 0;
            }
         }
      } else {
         for (j = 0; j < n; j++) {
            level_l[field_s]->data[j] = 0;
         }
      }
   }
}

/*
 * Calculates the source fields for a given level based on the data supplied by
 * restrict cycle
 */
void Hierarchy::calcSource(int lev) {
   int i, j, n;
   int field_u, field_s, field_res, field_jac;
   Mesh** level_l;
   Coords* coords_l;

   level_l = fields[lev];
   coords_l = coords[lev];
   for (i = 0; i < num_f; i++) {
      field_u = i + u_i * num_f;
      field_s = i + s_i * num_f;
      field_res = i + res_i * num_f;
      field_jac = i + jac_i * num_f;
      n = level_l[0]->n;

      if (n > MIN_GRID_SIZE) {
#         pragma omp parallel firstprivate(level_l, field_u, field_s, field_res, field_jac, n, j)
         {
#            pragma  omp for nowait
            for (j = 0; j < n; j++) {
               level_l[field_s]->data[j] += level_l[field_res]->data[j]
                       - level_l[field_jac]->data[j];
            }
         }
      } else {
         for (j = 0; j < n; j++) {
            level_l[field_s]->data[j] += level_l[field_res]->data[j]
                    - level_l[field_jac]->data[j];
         }
      }
   }
}

/*
 * Updates the fields based on data provided by interpolateCycle
 */
void Hierarchy::calcU(int lev) {
   int i, j, n;
   int field_u, field_s, field_res, field_jac;
   Mesh** level_l;
   Coords* coords_l;

   level_l = fields[lev];
   coords_l = coords[lev];
   for (i = 0; i < num_f; i++) {
      field_u = i + u_i * num_f;
      field_s = i + s_i * num_f;
      field_res = i + res_i * num_f;
      field_jac = i + jac_i * num_f;
      n = level_l[0]->n;

      if (n > MIN_GRID_SIZE) {
#         pragma omp parallel firstprivate(level_l, field_u, field_s, field_res, field_jac, n, j)
         {
#            pragma  omp for nowait
            for (j = 0; j < n; j++) {
               level_l[field_u]->data[j] += level_l[field_res]->data[j]
                       - level_l[field_jac]->data[j];
            }
         }
      } else {
         for (j = 0; j < n; j++) {
            level_l[field_u]->data[j] += level_l[field_res]->data[j]
                    - level_l[field_jac]->data[j];
         }
      }
   }
}

/*
 * Performs Jacobi relaxation of a single level
 *
 * itt:    number of smoothing cycles to apply
 * dt:     time step
 * lev:    level to relax
 * opp:    residual operator
 * jac:    jacobian operator
 */
void Hierarchy::relaxLevel(int itt, double dt, int lev,
        void (*opp)(double dt, Coords* coords, Mesh ** level),
        void (*jac)(double dt, Coords* coords, Mesh ** level)) {
   int i, j, k, n;
   int field_u, field_s, field_res, field_jac;
   Mesh** level_l;
   Coords* coords_l;

   level_l = fields[lev];
   coords_l = coords[lev];

   for (k = 0; k < itt; k++) {
      opp(dt, coords_l, level_l);
      s_resid(lev);
      jac(dt, coords_l, level_l);

      n = level_l[0]->n;


      for (i = 0; i < num_f; i++) {
         field_u = i + u_i * num_f;
         field_s = i + s_i * num_f;
         field_res = i + res_i * num_f;
         field_jac = i + jac_i * num_f;

         if (n > MIN_GRID_SIZE) {
#            pragma omp parallel firstprivate(level_l, field_u, field_s, \
            field_res, field_jac, j, n)
            {
#               pragma  omp for nowait
               for (j = 0; j < n; j++) {
                  level_l[field_u]->data[j] -= factor * level_l[field_res]->data[j]
                          / level_l[field_jac]->data[j];
               }
            }
         } else {
            for (j = 0; j < n; j++) {
               level_l[field_u]->data[j] -= factor * level_l[field_res]->data[j]
                       / level_l[field_jac]->data[j];
            }
         }
      }
   }
}

/*
 * Performs interpolate edge (coarse to fine) of the vCycle algorithm.
 * 
 * itt:    number of smoothing cycles to apply
 * dt:     time step
 * opp:    residual operator
 * jac:    jacobian operator
 */
void Hierarchy::interpolateCycle(int itt, double dt,
        void (*opp)(double dt, Coords* coords, Mesh ** level),
        void (*jac)(double dt, Coords* coords, Mesh ** level)) {
   int lev, i;
   int field_u, field_s, field_res, field_jac;
   Mesh* c;
   Mesh* f;
   Mesh** level_c;
   Mesh** level_f;
   Coords* coords_c;
   Coords* coords_f;

   for (lev = 1; lev < num_l; lev++) {
      
      level_c = fields[lev - 1];
      coords_c = coords[lev - 1];
      level_f = fields[lev];
      coords_f = coords[lev];
       
      for (i = 0; i < num_f; i++) {
         field_u = i + u_i * num_f;
         field_s = i + s_i * num_f;
         field_res = i + res_i * num_f;
         field_jac = i + jac_i * num_f;

         //interpolate u^{l-1} to res^{l}
         //I_{l-1}^{l} u^{l-1}
         c = level_c[field_u];
         f = level_f[field_res];
         Mesh::interpolate(c, f);
         
         //restrict u_l to res_l-1
         //I_{l}^{l-1} u^{l}
         c = level_c[field_res];
         f = level_f[field_u];
         Mesh::restrict(c, f);
         
         //restrict res_l-1 to jac_l
         //I_{l-1}^{l} I_{l}^{l-1} u^{l}
         c = level_c[field_res];
         f = level_f[field_jac];
         Mesh::interpolate(c, f);
      }
      calcU(lev);

      relaxLevel(itt, dt, lev, opp, jac);
   }
}

/*
 * Performs vCycle algorithm.
 * 
 * itt:    number of smoothing cycles to apply
 * dt:     time step
 * opp:    residual operator
 * jac:    jacobian operator
 */
void Hierarchy::vCycle(int itt, double dt,
        void (*opp)(double dt, Coords* coords, Mesh ** level),
        void (*jac)(double dt, Coords* coords, Mesh ** level)) {
   
   int i;
   zeroSource(num_l - 1);
   restrictCycle(itt, dt, opp, jac);
   relaxLevel(100, dt, 0, opp, jac);
   interpolateCycle(itt, dt, opp, jac);
   calcResidual();
   
}

/*
 * Calculates L2 norm of residual (provided by f_indep_resid) and the change
 * from the previous iteration.
 */
void Hierarchy::calcResidual()
{
   int i, j, k, n;
   int field_u, field_old, field_res;
   Mesh** level;
   double delta_u, residual;
   
   for (i = 0; i < num_l; i++)
   {
      delta_u = 0;
      residual = 0;
      n = fields[i][0]->n;
      level = fields[i];
      for (j = 0; j < num_f; j++)
      {
         field_u   = j + u_i * num_f;
         field_res = j + res_i * num_f;
         field_old = j + old_i * num_f;

         if (n > MIN_GRID_SIZE) {
#            pragma omp parallel shared(delta_u, residual) firstprivate(i, j, k, n, \
            field_u, field_old, field_res, level)
            {
#               pragma  omp for reduction(+:delta_u,residual)
               for (k = 0; k < n; k++) {
                  residual += pow(level[field_res]->data[k], 2.0);
                  delta_u += pow(level[field_old]->data[k] - level[field_u]->data[k], 2.0);
                  level[field_old]->data[k] = level[field_u]->data[k];
               }
            }
         } else {
            for (k = 0; k < n; k++) {
               residual += pow(level[field_res]->data[k], 2.0);
               delta_u += pow(level[field_old]->data[k] - level[field_u]->data[k], 2.0);
               level[field_old]->data[k] = level[field_u]->data[k];
            }
         }
      }
      change[i] = delta_u;
      change[i] /= (n*num_f);
      change[i] = sqrt(change[i]);
      
      resid[i] = residual;
      resid[i] /= (n*num_f);
      resid[i] = sqrt(resid[i]);
   }
}

/*
 * Prints a representation of level lev of the Hierarchy
 */
void Hierarchy::print(int lev) {
   int i;
   Mesh** level_l;
   Coords* coords_l;

   level_l = fields[lev];
   coords_l = coords[lev];
   printf("coordinates:\n");
   for (i = 0; i < dim; i++)
      Mesh::print(coords_l->mesh[i]);

   printf("fields:\n");
   for (i = 0; i < num_f; i++)
      Mesh::print(level_l[i + 0 * num_f]);
   printf("residuals:\n");
   for (i = 0; i < num_f; i++)
      Mesh::print(level_l[i + 1 * num_f]);
   printf("jacobians:\n");
   for (i = 0; i < num_f; i++)
      Mesh::print(level_l[i + 2 * num_f]);
   printf("sources:\n");
   for (i = 0; i < num_f; i++)
      Mesh::print(level_l[i + 3 * num_f]);
}

/*
 * Switches all pointers to fields filed1 and field2 on all levels. Important
 * for time stepping
 */
void Hierarchy::switchFields(int field1, int field2) {
   int i, j, k;
   for (i = 0; i < num_l; i++) {
      if (dim == 1) {
         double* temp;
         double* temp_data;
         temp = fields[i][field1]->f1d;
         fields[i][field1]->f1d = fields[i][field2]->f1d;
         fields[i][field2]->f1d = temp;
         
         temp_data = fields[i][field1]->data;
         fields[i][field1]->data = fields[i][field2]->data;
         fields[i][field2]->data = temp_data;
      } else if (dim == 2) {
         double** temp1;
         double* temp2;
         double* temp_data;
         int nx = fields[i][field1]->size[0];
         for (j = 0; j < nx; j++)
         {
            temp2 = fields[i][field1]->f2d[j];
            fields[i][field1]->f2d[j] = fields[i][field2]->f2d[j];
            fields[i][field2]->f2d[j] = temp2;
         }
         temp1 = fields[i][field1]->f2d;
         fields[i][field1]->f2d = fields[i][field2]->f2d;
         fields[i][field2]->f2d = temp1;
         
         temp_data = fields[i][field1]->data;
         fields[i][field1]->data = fields[i][field2]->data;
         fields[i][field2]->data = temp_data;
      } else if (dim == 3) {
         double*** temp1;
         double**  temp2;
         double*   temp3;
         double* temp_data;
         int nx = fields[i][field1]->size[0];
         int ny = fields[i][field1]->size[1];
         for (j = 0; j < nx; j++)
         {
            for (k = 0; k < ny; k++)
            {
               temp3 = fields[i][field1]->f3d[j][k];
               fields[i][field1]->f3d[j][k] = fields[i][field2]->f3d[j][k];
               fields[i][field2]->f3d[j][k] = temp3;   
            }
            temp2 = fields[i][field1]->f3d[j];
            fields[i][field1]->f3d[j] = fields[i][field2]->f3d[j];
            fields[i][field2]->f3d[j] = temp2;
         }
         temp1 = fields[i][field1]->f3d;
         fields[i][field1]->f3d = fields[i][field2]->f3d;
         fields[i][field2]->f3d = temp1;
         
         temp_data = fields[i][field1]->data;
         fields[i][field1]->data = fields[i][field2]->data;
         fields[i][field2]->data = temp_data;
      }
   }
}

/*
 * Repeated operation in relaxation operation. Subtracts source fields from 
 * residual.
 */
void Hierarchy::s_resid(int lev) {

   Mesh** level;
   int i, j, n;
   level = fields[lev];
   n = level[0]-> n;

   if (n > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(i,j,n,level)
      {
         int field_s = num_f*s_i;
         int field_res = num_f*res_i;
#         pragma  omp for nowait
         for (i = 0; i < n; i++) {
            for (j = 0; j < num_f; j++) {
               level[j + field_res]->data[i] -= level[j + field_s]->data[i];
            }
         }
      }
   } else {
      int field_s = num_f*s_i;
      int field_res = num_f*res_i;
#      pragma  omp for nowait
      for (i = 0; i < n; i++) {
         for (j = 0; j < num_f; j++) {
            level[j + field_res]->data[i] -= level[j + field_s]->data[i];
         }
      }
   }
}