/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */


/** \example amg.cpp
*
*   This tutorial shows the use of algebraic multigrid (AMG) preconditioners.
*   \warning AMG is currently only experimentally available with the OpenCL backend and depends on Boost.uBLAS
*
*   We start with some rather general includes and preprocessor variables:
**/


#include "viennacl/vector.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/tools/timer.hpp"

/**
* Import the AMG functionality:
**/
#include "viennacl/linalg/amg.hpp"

/**
* Some more includes:
**/
#include <iostream>
#include <vector>
#include <ctime>


/** <h2>Part 1: Worker routines</h2>
*
*  <h3>Run the Solver</h3>
*   Runs the provided solver specified in the `solver` object with the provided preconditioner `precond`
**/
template<typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag>
void run_solver(MatrixType const & matrix, VectorType const & rhs, VectorType const & ref_result, SolverTag const & solver, PrecondTag const & precond)
{
  VectorType result(rhs);
  VectorType residual(rhs);

  result = viennacl::linalg::solve(matrix, rhs, solver, precond);
  residual -= viennacl::linalg::prod(matrix, result);
  std::cout << "  > Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(rhs) << std::endl;
  std::cout << "  > Iterations: " << solver.iters() << std::endl;
  result -= ref_result;
  std::cout << "  > Relative deviation from result: " << viennacl::linalg::norm_2(result) / viennacl::linalg::norm_2(ref_result) << std::endl;
}


void run_solver(std::string const & filename, viennacl::context const & ctx)
{
  typedef double    ScalarType;  // feel free to change this to double if supported by your device

  viennacl::compressed_matrix<ScalarType> vcl_compressed_matrix(ctx);

  std::vector< std::map<unsigned int, ScalarType> > read_in_matrix;
  if (!viennacl::io::read_matrix_market_file(read_in_matrix, filename))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    exit(EXIT_FAILURE);
  }
  viennacl::copy(read_in_matrix, vcl_compressed_matrix);

  viennacl::vector<ScalarType> vcl_vec(vcl_compressed_matrix.size1(), ctx);
  viennacl::vector<ScalarType> vcl_ref_result(vcl_compressed_matrix.size1(), ctx);

  std::vector<ScalarType> std_vec, std_result;

  // rhs and result vector:
  std_vec.resize(vcl_compressed_matrix.size1());
  std_result.resize(vcl_compressed_matrix.size1());
  for (std::size_t i=0; i<std_result.size(); ++i)
    std_result[i] = ScalarType(1);

  // Copy to GPU
  viennacl::copy(std_vec, vcl_vec);
  viennacl::copy(std_result, vcl_ref_result);

  vcl_vec = viennacl::linalg::prod(vcl_compressed_matrix, vcl_ref_result);

  viennacl::linalg::cg_tag cg_solver;

  /**
  * Generate the setup for an AMG preconditioner:

  amg_tag = viennacl::linalg::amg_tag(VIENNACL_AMG_COARSE_RS,       // coarsening strategy
                                      VIENNACL_AMG_INTERPOL_DIRECT, // interpolation strategy
                                      0.25, // strength of dependence threshold
                                      0.2,  // interpolation weight
                                      0.67, // jacobi smoother weight
                                      3,    // presmoothing steps
                                      3,    // postsmoothing steps
                                      0);   // number of coarse levels to be used (0: automatically use as many as reasonable)
  */

  /**
  * Generate the setup for an AMG preconditioner which as aggregation-based (AG)
  **/
  viennacl::linalg::amg_tag amg_tag;
  amg_tag.set_coarsening_method(viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION);
  amg_tag.set_interpolation_method(viennacl::linalg::AMG_INTERPOLATION_METHOD_AGGREGATION);

  viennacl::linalg::amg_precond<viennacl::compressed_matrix<ScalarType> > vcl_amg = viennacl::linalg::amg_precond<viennacl::compressed_matrix<ScalarType> > (vcl_compressed_matrix, amg_tag);
  vcl_amg.setup();
  viennacl::vector<ScalarType> vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec, cg_solver, vcl_amg);


  viennacl::tools::timer timer;
  viennacl::linalg::amg_precond<viennacl::compressed_matrix<ScalarType> > vcl_amg2 = viennacl::linalg::amg_precond<viennacl::compressed_matrix<ScalarType> > (vcl_compressed_matrix, amg_tag);
  timer.start();
  vcl_amg2.setup();
  vcl_result = viennacl::linalg::solve(vcl_compressed_matrix, vcl_vec, cg_solver, vcl_amg2);
  viennacl::vector<ScalarType> vcl_residual(vcl_vec);
  vcl_residual -= viennacl::linalg::prod(vcl_compressed_matrix, vcl_result);
  std::cout << vcl_vec.size() << "      ";
  std::cout << timer.get() << "     ";
  std::cout << cg_solver.iters() << "       ";
  std::cout << viennacl::linalg::norm_2(vcl_residual) / viennacl::linalg::norm_2(vcl_vec) << std::endl;
}



/**
*  <h2>Part 2: Run Solvers with AMG Preconditioners</h2>
*
*  In this
**/
int main()
{
  viennacl::context ctx;

  std::cout << "# Size     Time       Iters       RelResidual" << std::endl;
  run_solver("./data/poisson2d_3969.mtx", ctx);
  run_solver("./data/poisson2d_16129.mtx", ctx);
  run_solver("./data/poisson2d_65025.mtx", ctx);
  run_solver("./data/poisson2d_261121.mtx", ctx);


  return EXIT_SUCCESS;
}

