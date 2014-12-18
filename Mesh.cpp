#include "Mesh.hpp"
#include <stdio.h>
#include <omp.h>

/*
 * Represents a field in a Hiearchy instance.
 * dim_:  dimension of the Hiearchy (1,2,3)
 * size_: array of length dim_ containing the number of grid points in each 
 *        dimension
 */
Mesh::Mesh(int dim_, int* size_) {
   int i, j;
   dim = dim_;
   n = 1;
   for (i = 0; i < dim; i++) {
      size[i] = size_[i];
      n *= size[i];
   }
   data = new double[n];

   if (dim == 1) {
      f1d = data;
   } else if (dim == 2) {
      f2d = new double*[size[0]];
      for (i = 0; i < size[0]; i++) {
         f2d[i] = &(data[i * size[1]]);
      }
   } else if (dim == 3) {
      f3d = new double**[size[0]];
      for (i = 0; i < size[0]; i++) {
         f3d[i] = new double*[size[1]];
         for (j = 0; j < size[1]; j++) {
            f3d[i][j] = &(data[i * (size[1] * size[2]) + j * size[2]]);
         }
      }
   }
}

Mesh::~Mesh() {
   int i;
   if (dim == 2) {
      delete [] f2d;
   } else if (dim == 3) {
      for (i = 0; i < size[0]; i++) {
         delete [] f3d[i];
      }
      delete [] f3d;
   }
   delete [] data;
}

/*
 * Prints out a 1d representation of the Mesh. Defined only if the Mesh is 1 
 * dimensional. 
 */
void Mesh::print1d(Mesh* mesh) {
   int i;
   for (i = 0; i < mesh->size[0]; i++) {
      printf("%10.6lf ", mesh->f1d[i]);
   }
   printf("\n\n");
}

/*
 * Prints out a 2d representation of the Mesh. Defined only if the Mesh is 2 
 * dimensional. 
 */
void Mesh::print2d(Mesh* mesh) {
   int i, j;
   for (j = 0; j < mesh->size[1]; j++) {
      for (i = 0; i < mesh->size[0]; i++) {
         printf("%10.6lf ", mesh->f2d[i][j]);
      }
      printf("\n");
   }
   printf("\n\n");
}

/*
 * Prints out a 3d representation of the Mesh. Defined only if the Mesh is 3 
 * dimensional. 
 */
void Mesh::print3d(Mesh* mesh) {
   int i, j, k;
   for (k = 0; k < mesh->size[2]; k++) {
      printf("k = %d slice\n", k);
      for (j = 0; j < mesh->size[1]; j++) {
         for (i = 0; i < mesh->size[0]; i++) {
            printf("%10.6lf ", mesh->f3d[i][j][k]);
         }
         printf("\n");
      }
      printf("\n\n");
   }
   printf("\n\n");
}

/*
 * Prints out a representation of the Mesh based on its dimension.
 */
void Mesh::print(Mesh* mesh) {
   if (mesh->dim == 1) {
      print1d(mesh);
   } else if (mesh -> dim == 2) {
      print2d(mesh);
   } else if (mesh -> dim == 3) {
      print3d(mesh);
   }
}

/*
 * Restricts a fine mesh of edge dimension n to a coarse mesh of edge dimension
 * n/2+1. Uses the stencil: 
 * 
 * [0.25 0.5 0.25]
 */
int Mesh::restrict1d(Mesh* c, Mesh* f) {
   int i, idi;
   int c_nx;
   int f_nx;
   c_nx = c->size[0];
   f_nx = f->size[0];

   if (f_nx != 2 * c_nx - 1) {
      printf("Incorrect dimensions on grids for interpolation\n");
      printf("Course grid is (%d) while fine grid is (%d)\n",
              c_nx, f_nx);
      printf("Require (%d) for fine grid\n", 2 * c_nx - 1);
      return 0;
   }

   if (f->n > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(c, f, c_nx, f_nx, i, idi)
      {
#         pragma  omp for nowait
         for (i = 0; i < c_nx; i++) {
            idi = 2 * i;
            if (i != 0 && i != c_nx - 1) {
               c->f1d[i] = 0.5 * f->f1d[idi]
                       + 0.25 * (f->f1d[idi + 1] + f->f1d[idi - 1]);
            } else {
               c->f1d[i] = f->f1d[idi];
            }
         }
      }
   } else {
      for (i = 0; i < c_nx; i++) {
         idi = 2 * i;
         if (i != 0 && i != c_nx - 1) {
            c->f1d[i] = 0.5 * f->f1d[idi]
                    + 0.25 * (f->f1d[idi + 1] + f->f1d[idi - 1]);
         } else {
            c->f1d[i] = f->f1d[idi];
         }
      }
   }
   return 1;
}

/*
 * Restricts a fine mesh of edge dimension n to a coarse mesh of edge dimension
 * n/2+1. Uses the stencil:
 * 
 * [0.0625  0.125  0.0625]
 * [0.125   0.25   0.125 ]
 * [0.0625  0.125  0.0625]
 */
int Mesh::restrict2d(Mesh* c, Mesh* f) {
   int i, j, idi, idj;
   int c_nx, c_ny;
   int f_nx, f_ny;
   c_nx = c->size[0];
   c_ny = c->size[1];
   f_nx = f->size[0];
   f_ny = f->size[1];

   if (f_nx != 2 * c_nx - 1 || f_ny != 2 * c_ny - 1) {
      printf("Incorrect dimensions on grids for interpolation\n");
      printf("Course grid is (%d, %d) while fine grid is (%d, %d)\n",
              c_nx, c_ny, f_nx, f_ny);
      printf("Require (%d, %d) for fine grid\n", 2 * c_nx - 1, 2 * c_ny - 1);
      return 0;
   }

   if (f->n > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(c, f, c_nx, c_ny, f_nx, f_ny, i, idi, j, idj)
      {
#         pragma  omp for nowait
         for (i = 0; i < c_nx; i++) {
            for (j = 0; j < c_ny; j++) {
               idi = 2 * i;
               idj = 2 * j;
               if (i != 0 && i != c_nx - 1 && j != 0 && j != c_ny - 1) {
                  c->f2d[i][j] = 0.25 * f->f2d[idi][idj]
                          + 0.125 * (f->f2d[idi + 1][idj] + f->f2d[idi - 1][idj]
                          + f->f2d[idi][idj + 1] + f->f2d[idi][idj - 1])
                          + 0.0625 * (f->f2d[idi + 1][idj + 1]
                          + f->f2d[idi + 1][idj - 1] + f->f2d[idi - 1][idj + 1]
                          + f->f2d[idi - 1][idj - 1]);
               } else {
                  c->f2d[i][j] = f->f2d[idi][idj];
               }
            }
         }
      }
   } else {
      for (i = 0; i < c_nx; i++) {
         for (j = 0; j < c_ny; j++) {
            idi = 2 * i;
            idj = 2 * j;
            if (i != 0 && i != c_nx - 1 && j != 0 && j != c_ny - 1) {
               c->f2d[i][j] = 0.25 * f->f2d[idi][idj]
                       + 0.125 * (f->f2d[idi + 1][idj] + f->f2d[idi - 1][idj]
                       + f->f2d[idi][idj + 1] + f->f2d[idi][idj - 1])
                       + 0.0625 * (f->f2d[idi + 1][idj + 1]
                       + f->f2d[idi + 1][idj - 1] + f->f2d[idi - 1][idj + 1]
                       + f->f2d[idi - 1][idj - 1]);
            } else {
               c->f2d[i][j] = f->f2d[idi][idj];
            }
         }
      }
   }
   return 1;
}

/*
 * Restricts a fine mesh of edge dimension n to a coarse mesh of edge dimension
 * n/2+1. Uses the stencil:
 * 
 * k=-1
 * [0.015625  0.03125  0.015625]
 * [0.03125   0.0625   0.03125 ]
 * [0.015625  0.03125  0.015625]
 * 
 * k=0
 * [0.03125   0.0625   0.03125]
 * [0.0625    0.125    0.0625 ]
 * [0.03125   0.0625   0.03125]
 * 
 * k=1
 * [0.015625  0.03125  0.015625]
 * [0.03125   0.0625   0.03125 ]
 * [0.015625  0.03125  0.015625]
 */
int Mesh::restrict3d(Mesh* c, Mesh* f) {
   int i, j, k, idi, idj, idk;
   int c_nx, c_ny, c_nz;
   int f_nx, f_ny, f_nz;
   c_nx = c->size[0];
   c_ny = c->size[1];
   c_nz = c->size[2];
   f_nx = f->size[0];
   f_ny = f->size[1];
   f_nz = f->size[2];

   if (f_nx != 2 * c_nx - 1 || f_ny != 2 * c_ny - 1 || f_nz != 2 * c_nz - 1) {
      printf("Incorrect dimensions on grids for interpolation\n");
      printf("Course grid is (%d, %d, %d) while fine grid is (%d, %d, %d)\n",
              c_nx, c_ny, c_nz, f_nx, f_ny, f_nz);
      printf("Require (%d, %d, %d) for fine grid\n",
              2 * c_nx - 1, 2 * c_ny - 1, 2 * c_nz - 1);
      return 0;
   }

   if (f->n > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(c, f, c_nx, c_ny, c_nz, f_nx, f_ny, f_nz, i, idi, j, idj, k, idk)
      {
#         pragma  omp for nowait
         for (i = 0; i < c_nx; i++) {
            for (j = 0; j < c_ny; j++) {
               for (k = 0; k < c_nz; k++) {
                  idi = 2 * i;
                  idj = 2 * j;
                  idk = 2 * k;
                  if (i != 0 && i != c_nx - 1 && j != 0 && j != c_ny - 1 && k != 0
                          && k != c_ny - 1) {
                     c->f3d[i][j][k] = 0.125 * f->f3d[idi][idj][idk]
                             + 0.0625 * (f->f3d[idi + 1][idj][idk]
                             + f->f3d[idi - 1][idj][idk] + f->f3d[idi][idj + 1][idk]
                             + f->f3d[idi][idj - 1][idk] + f->f3d[idi][idj][idk + 1]
                             + f->f3d[idi][idj][idk - 1])
                             + 0.03125 * (f->f3d[idi + 1][idj][idk + 1]
                             + f->f3d[idi - 1][idj][idk + 1] + f->f3d[idi][idj + 1][idk + 1]
                             + f->f3d[idi][idj - 1][idk + 1] + f->f3d[idi + 1][idj][idk - 1]
                             + f->f3d[idi - 1][idj][idk - 1] + f->f3d[idi][idj + 1][idk - 1]
                             + f->f3d[idi][idj - 1][idk - 1] + f->f3d[idi + 1][idj + 1][idk]
                             + f->f3d[idi - 1][idj + 1][idk] + f->f3d[idi + 1][idj - 1][idk]
                             + f->f3d[idi - 1][idj - 1][idk])
                             + 0.015625 * (f->f3d[idi + 1][idj + 1][idk + 1]
                             + f->f3d[idi + 1][idj + 1][idk - 1] + f->f3d[idi + 1][idj - 1][idk + 1]
                             + f->f3d[idi + 1][idj - 1][idk - 1] + f->f3d[idi - 1][idj + 1][idk + 1]
                             + f->f3d[idi - 1][idj + 1][idk - 1] + f->f3d[idi - 1][idj - 1][idk + 1]
                             + f->f3d[idi - 1][idj - 1][idk - 1]);

                  } else {
                     c->f3d[i][j][k] = f->f3d[idi][idj][idk];
                  }
               }
            }
         }
      }
   } else {
      for (i = 0; i < c_nx; i++) {
         for (j = 0; j < c_ny; j++) {
            for (k = 0; k < c_nz; k++) {
               idi = 2 * i;
               idj = 2 * j;
               idk = 2 * k;
               if (i != 0 && i != c_nx - 1 && j != 0 && j != c_ny - 1 && k != 0
                       && k != c_ny - 1) {
                  c->f3d[i][j][k] = 0.125 * f->f3d[idi][idj][idk]
                          + 0.0625 * (f->f3d[idi + 1][idj][idk]
                          + f->f3d[idi - 1][idj][idk] + f->f3d[idi][idj + 1][idk]
                          + f->f3d[idi][idj - 1][idk] + f->f3d[idi][idj][idk + 1]
                          + f->f3d[idi][idj][idk - 1])
                          + 0.03125 * (f->f3d[idi + 1][idj][idk + 1]
                          + f->f3d[idi - 1][idj][idk + 1] + f->f3d[idi][idj + 1][idk + 1]
                          + f->f3d[idi][idj - 1][idk + 1] + f->f3d[idi + 1][idj][idk - 1]
                          + f->f3d[idi - 1][idj][idk - 1] + f->f3d[idi][idj + 1][idk - 1]
                          + f->f3d[idi][idj - 1][idk - 1] + f->f3d[idi + 1][idj + 1][idk]
                          + f->f3d[idi - 1][idj + 1][idk] + f->f3d[idi + 1][idj - 1][idk]
                          + f->f3d[idi - 1][idj - 1][idk])
                          + 0.015625 * (f->f3d[idi + 1][idj + 1][idk + 1]
                          + f->f3d[idi + 1][idj + 1][idk - 1] + f->f3d[idi + 1][idj - 1][idk + 1]
                          + f->f3d[idi + 1][idj - 1][idk - 1] + f->f3d[idi - 1][idj + 1][idk + 1]
                          + f->f3d[idi - 1][idj + 1][idk - 1] + f->f3d[idi - 1][idj - 1][idk + 1]
                          + f->f3d[idi - 1][idj - 1][idk - 1]);

               } else {
                  c->f3d[i][j][k] = f->f3d[idi][idj][idk];
               }
            }
         }
      }
   }
   return 1;
}

/*
 * Restricts a fine mesh of edge dimension n to a coarse mesh of edge dimension
 * n/2+1. Uses the stencil:
 */
int Mesh::restrict(Mesh* c, Mesh* f) {
   if (c->dim == 1 && f->dim == 1) {
      return restrict1d(c, f);
   } else if (c -> dim == 2 && f->dim == 2) {
      return restrict2d(c, f);
   } else if (c -> dim == 3 && f->dim == 3) {
      return restrict3d(c, f);
   } else {
      printf("Restriction could not proceed as dimension of the coarse mesh "
              "does not match the dimension of the fine mesh\n");
      return 0;
   }
}

/*
 * Interpolates a coarse mesh of edge dimension n to a fine mesh of edge 
 * dimension 2*n-1. Uses even weighted linear interpolation:
 */
int Mesh::interpolate1d(Mesh* c, Mesh* f) {
   int i, ii, idi;
   int c_nx;
   int f_nx;
   c_nx = c->size[0];
   f_nx = f->size[0];

   if (f_nx != 2 * c_nx - 1) {
      printf("Incorrect dimensions on grids for interpolation\n");
      printf("Course grid is (%d) while fine grid is (%d)\n",
              c_nx, f_nx);
      printf("Require (%d) for fine grid\n", 2 * c_nx - 1);
      return 0;
   }

   if (f->n > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(c, f, c_nx, f_nx, i, ii, idi)
      {
#         pragma  omp for nowait
         for (i = 0; i < c_nx; i++) {
            for (ii = 0; ii < 2; ii++) {
               idi = 2 * i + ii;
               if (idi < f_nx) {
                  if (ii == 0)
                     f->f1d[idi] = c->f1d[i];
                  else if (ii == 1)
                     f->f1d[idi] = 0.5 * (c->f1d[i] + c->f1d[i + 1]);
               }
            }
         }
      }
   } else {
      for (i = 0; i < c_nx; i++) {
         for (ii = 0; ii < 2; ii++) {
            idi = 2 * i + ii;
            if (idi < f_nx) {
               if (ii == 0)
                  f->f1d[idi] = c->f1d[i];
               else if (ii == 1)
                  f->f1d[idi] = 0.5 * (c->f1d[i] + c->f1d[i + 1]);
            }
         }
      }
   }
   return 1;
}

/*
 * Interpolates a coarse mesh of edge dimension n to a fine mesh of edge 
 * dimension 2*n-1. Uses even weighted linear interpolation:
 */
int Mesh::interpolate2d(Mesh* c, Mesh* f) {
   int i, j, ii, jj, idi, idj;
   int c_nx, c_ny;
   int f_nx, f_ny;
   c_nx = c->size[0];
   c_ny = c->size[1];
   f_nx = f->size[0];
   f_ny = f->size[1];

   if (f_nx != 2 * c_nx - 1 || f_ny != 2 * c_ny - 1) {
      printf("Incorrect dimensions on grids for interpolation\n");
      printf("Course grid is (%d, %d) while fine grid is (%d, %d)\n",
              c_nx, c_ny, f_nx, f_ny);
      printf("Require (%d, %d) for fine grid\n", 2 * c_nx - 1, 2 * c_ny - 1);
      return 0;
   }

   if (f->n > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(c, f, c_nx, c_ny, f_nx, f_ny, i, ii, idi, j, jj, idj)
      {
#         pragma  omp for nowait
         for (i = 0; i < c_nx; i++) {
            for (j = 0; j < c_ny; j++) {
               for (ii = 0; ii < 2; ii++) {
                  for (jj = 0; jj < 2; jj++) {
                     idi = 2 * i + ii;
                     idj = 2 * j + jj;
                     if (idi < f_nx && idj < f_ny) {
                        if (ii == 0 && jj == 0)
                           f->f2d[idi][idj] = c->f2d[i][j];
                        else if (ii == 0 && jj == 1)
                           f->f2d[idi][idj] = 0.5 * (c->f2d[i][j] + c->f2d[i][j + 1]);
                        else if (ii == 1 && jj == 0)
                           f->f2d[idi][idj] = 0.5 * (c->f2d[i][j] + c->f2d[i + 1][j]);
                        else if (ii == 1 && jj == 1)
                           f->f2d[idi][idj] = 0.25 * (c->f2d[i][j] + c->f2d[i + 1][j]
                                + c->f2d[i][j + 1] + c->f2d[i + 1][j + 1]);
                     }
                  }
               }
            }
         }
      }
   } else {
      for (i = 0; i < c_nx; i++) {
         for (j = 0; j < c_ny; j++) {
            for (ii = 0; ii < 2; ii++) {
               for (jj = 0; jj < 2; jj++) {
                  idi = 2 * i + ii;
                  idj = 2 * j + jj;
                  if (idi < f_nx && idj < f_ny) {
                     if (ii == 0 && jj == 0)
                        f->f2d[idi][idj] = c->f2d[i][j];
                     else if (ii == 0 && jj == 1)
                        f->f2d[idi][idj] = 0.5 * (c->f2d[i][j] + c->f2d[i][j + 1]);
                     else if (ii == 1 && jj == 0)
                        f->f2d[idi][idj] = 0.5 * (c->f2d[i][j] + c->f2d[i + 1][j]);
                     else if (ii == 1 && jj == 1)
                        f->f2d[idi][idj] = 0.25 * (c->f2d[i][j] + c->f2d[i + 1][j]
                             + c->f2d[i][j + 1] + c->f2d[i + 1][j + 1]);
                  }
               }
            }
         }
      }
   }
   return 1;
}

/*
 * Interpolates a coarse mesh of edge dimension n to a fine mesh of edge 
 * dimension 2*n-1. Uses even weighted linear interpolation:
 */
int Mesh::interpolate3d(Mesh* c, Mesh* f) {
   int i, j, k, ii, jj, kk, idi, idj, idk;
   int c_nx, c_ny, c_nz;
   int f_nx, f_ny, f_nz;
   c_nx = c->size[0];
   c_ny = c->size[1];
   c_nz = c->size[2];
   f_nx = f->size[0];
   f_ny = f->size[1];
   f_nz = f->size[2];

   if (f_nx != 2 * c_nx - 1 || f_ny != 2 * c_ny - 1 || f_nz != 2 * c_nz - 1) {
      printf("Incorrect dimensions on grids for interpolation\n");
      printf("Course grid is (%d, %d, %d) while fine grid is (%d, %d, %d)\n",
              c_nx, c_ny, c_nz, f_nx, f_ny, f_nz);
      printf("Require (%d, %d, %d) for fine grid\n",
              2 * c_nx - 1, 2 * c_ny - 1, 2 * c_nz - 1);
      return 0;
   }

   if (f->n > MIN_GRID_SIZE) {
#      pragma omp parallel firstprivate(c, f, c_nx, c_ny, c_nz, f_nx, f_ny, f_nz, i, ii, idi, j, jj, idj, k, kk, idk)
      {
#         pragma  omp for nowait
         for (i = 0; i < c_nx; i++) {
            for (j = 0; j < c_ny; j++) {
               for (k = 0; k < c_nz; k++) {
                  for (ii = 0; ii < 2; ii++) {
                     for (jj = 0; jj < 2; jj++) {
                        for (kk = 0; kk < 2; kk++) {
                           idi = 2 * i + ii;
                           idj = 2 * j + jj;
                           idk = 2 * k + kk;
                           if (idi < f_nx && idj < f_ny && idk < f_nz) {
                              if (ii == 0 && jj == 0 && kk == 0)
                                 f->f3d[idi][idj][idk] = c->f3d[i][j][k];
                              else if (ii == 0 && jj == 0 && kk == 1)
                                 f->f3d[idi][idj][idk] = 0.5 * (c->f3d[i][j][k]
                                      + c->f3d[i][j][k + 1]);
                              else if (ii == 0 && jj == 1 && kk == 0)
                                 f->f3d[idi][idj][idk] = 0.5 * (c->f3d[i][j][k]
                                      + c->f3d[i][j + 1][k]);
                              else if (ii == 0 && jj == 1 && kk == 1)
                                 f->f3d[idi][idj][idk] = 0.25 * (c->f3d[i][j][k]
                                      + c->f3d[i][j + 1][k] + c->f3d[i][j][k + 1]
                                      + c->f3d[i][j + 1][k + 1]);
                              else if (ii == 1 && jj == 0 && kk == 0)
                                 f->f3d[idi][idj][idk] = 0.5 * (c->f3d[i][j][k]
                                      + c->f3d[i + 1][j][k]);
                              else if (ii == 1 && jj == 0 && kk == 1)
                                 f->f3d[idi][idj][idk] = 0.25 * (c->f3d[i][j][k]
                                      + c->f3d[i + 1][j][k] + c->f3d[i][j][k + 1]
                                      + c->f3d[i + 1][j][k + 1]);
                              else if (ii == 1 && jj == 1 && kk == 0)
                                 f->f3d[idi][idj][idk] = 0.25 * (c->f3d[i][j][k]
                                      + c->f3d[i][j + 1][k] + c->f3d[i + 1][j][k]
                                      + c->f3d[i + 1][j + 1][k]);
                              else if (ii == 1 && jj == 1 && kk == 1)
                                 f->f3d[idi][idj][idk] = 0.125 * (c->f3d[i][j][k]
                                      + c->f3d[i][j][k + 1] + c->f3d[i][j + 1][k]
                                      + c->f3d[i][j + 1][k + 1] + c->f3d[i + 1][j][k]
                                      + c->f3d[i + 1][j][k + 1] + c->f3d[i + 1][j + 1][k]
                                      + c->f3d[i + 1][j + 1][k + 1]);
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   } else {
      for (i = 0; i < c_nx; i++) {
         for (j = 0; j < c_ny; j++) {
            for (k = 0; k < c_nz; k++) {
               for (ii = 0; ii < 2; ii++) {
                  for (jj = 0; jj < 2; jj++) {
                     for (kk = 0; kk < 2; kk++) {
                        idi = 2 * i + ii;
                        idj = 2 * j + jj;
                        idk = 2 * k + kk;
                        if (idi < f_nx && idj < f_ny && idk < f_nz) {
                           if (ii == 0 && jj == 0 && kk == 0)
                              f->f3d[idi][idj][idk] = c->f3d[i][j][k];
                           else if (ii == 0 && jj == 0 && kk == 1)
                              f->f3d[idi][idj][idk] = 0.5 * (c->f3d[i][j][k]
                                   + c->f3d[i][j][k + 1]);
                           else if (ii == 0 && jj == 1 && kk == 0)
                              f->f3d[idi][idj][idk] = 0.5 * (c->f3d[i][j][k]
                                   + c->f3d[i][j + 1][k]);
                           else if (ii == 0 && jj == 1 && kk == 1)
                              f->f3d[idi][idj][idk] = 0.25 * (c->f3d[i][j][k]
                                   + c->f3d[i][j + 1][k] + c->f3d[i][j][k + 1]
                                   + c->f3d[i][j + 1][k + 1]);
                           else if (ii == 1 && jj == 0 && kk == 0)
                              f->f3d[idi][idj][idk] = 0.5 * (c->f3d[i][j][k]
                                   + c->f3d[i + 1][j][k]);
                           else if (ii == 1 && jj == 0 && kk == 1)
                              f->f3d[idi][idj][idk] = 0.25 * (c->f3d[i][j][k]
                                   + c->f3d[i + 1][j][k] + c->f3d[i][j][k + 1]
                                   + c->f3d[i + 1][j][k + 1]);
                           else if (ii == 1 && jj == 1 && kk == 0)
                              f->f3d[idi][idj][idk] = 0.25 * (c->f3d[i][j][k]
                                   + c->f3d[i][j + 1][k] + c->f3d[i + 1][j][k]
                                   + c->f3d[i + 1][j + 1][k]);
                           else if (ii == 1 && jj == 1 && kk == 1)
                              f->f3d[idi][idj][idk] = 0.125 * (c->f3d[i][j][k]
                                   + c->f3d[i][j][k + 1] + c->f3d[i][j + 1][k]
                                   + c->f3d[i][j + 1][k + 1] + c->f3d[i + 1][j][k]
                                   + c->f3d[i + 1][j][k + 1] + c->f3d[i + 1][j + 1][k]
                                   + c->f3d[i + 1][j + 1][k + 1]);
                        }
                     }
                  }
               }
            }
         }
      }
   }
   return 1;
}

/*
 * Interpolates a coarse mesh of edge dimension n to a fine mesh of edge 
 * dimension 2*n-1. Uses even weighted linear interpolation:
 */
int Mesh::interpolate(Mesh* c, Mesh* f) {
   if (c->dim == 1 && f->dim == 1) {
      return interpolate1d(c, f);
   } else if (c -> dim == 2 && f->dim == 2) {
      return interpolate2d(c, f);
   } else if (c -> dim == 3 && f->dim == 3) {
      return interpolate3d(c, f);
   } else {
      printf("Interpolation could not proceed as dimension of the coarse mesh "
              "does not match the dimension of the fine mesh\n");
      return 0;
   }
}