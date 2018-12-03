/* Copyright (C) 2018 The LinBox group
 * Written by Hongguang Zhu <zhuhongguang2014@gmail.com>
 *
 * ========LICENCE========
 * This file is part of the library LinBox.
 *
 * LinBox is free software: you can redistribute it and/or modify
 * it under the terms of the  GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 * ========LICENCE========
 */

/*! @file tests/test-solveCRA.C
 * @ingroup benchmarks
 * @brief Testing the MPI parallel/serial rational solver
 */

#include "givaro/modular.h"
#include "givaro/zring.h"
#include "linbox/linbox-config.h"
#include "linbox/matrix/sparse-matrix.h"
#include "linbox/matrix/random-matrix.h"
#include "linbox/solutions/methods.h"
#include "linbox/solutions/solve.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#ifdef __LINBOX_HAVE_MPI
#include "linbox/util/mpicpp.h"
#include <mpi.h>
#else
#include "linbox/algorithms/cra-domain-omp.h" //<---Only compile without MPI
#endif

using namespace LinBox;

template <class Field, class Matrix>
static bool checkResult(const Field& ZZ, Matrix& A, BlasVector<Field>& B, BlasVector<Field>& X, Integer& d)
{
    BlasVector<Field> B2(ZZ, A.coldim());
    BlasVector<Field> B3(ZZ, A.coldim());
    A.apply(B2, X);

    Integer tmp;
    for (size_t j = 0; j < B.size(); ++j) {
        B3.setEntry(j, d * B.getEntry(j));
    }
    for (size_t j = 0; j < A.coldim(); ++j) {
        if (!ZZ.areEqual(B2[j], B3[j])) {
            std::cerr << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
            std::cerr << "               The solution of solveCRA is incorrect                " << std::endl;
            std::cerr << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
            return false;
        }
    }
    return true;
}

template <class Field, class Matrix>
void genData(Field& F, Matrix& A, size_t bits)
{
    typedef typename Field::RandIter RandIter;
    RandIter RI(F, bits, 6);
    LinBox::RandomDenseMatrix<RandIter, Field> RDM(F, RI);
    RDM.randomFullRank(A);
}

template <class Field>
void genData(Field& F, BlasVector<Field>& B, size_t bits)
{
    typedef typename Field::RandIter RandIter;
    RandIter RI(F, bits, 6);
    B.random(RI);
}


bool test_set(BlasVector<Givaro::ZRing<Integer>>& X2, BlasMatrix<Givaro::ZRing<Integer>>& A,
              BlasVector<Givaro::ZRing<Integer>>& B, Communicator* Cptr)
{
    bool tag = false;
    Givaro::ZRing<Integer> ZZ;
    Givaro::ZRing<Integer>::Element d;

    // ----- Results verification

    RingCategories::IntegerTag tg;

#ifdef __LINBOX_HAVE_MPI
    double starttime = MPI_Wtime();
    solveCRA(X2, d, A, B, tg, Method::BlasElimination(), Cptr);
    double endtime = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
#else
    double starttime = omp_get_wtime();
    solveCRA(X2, d, A, B, tg, Method::BlasElimination());
    double endtime = omp_get_wtime();
#endif

    if (Cptr->master()) {
        std::cout << "Total CPU time (seconds): " << endtime - starttime << std::endl;
        tag = checkResult(ZZ, A, B, X2, d);
    }

#ifdef __LINBOX_HAVE_MPI
    MPI_Bcast(&tag, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
#endif
    return tag;
}

int main(int argc, char** argv)
{
    Communicator communicator(&argc, &argv);

    size_t bits, niter, ni, nj;
    bits = 10, niter = 1, ni = 1, nj = 1;

    static Argument args[] = {{'n', "-n N", "Set column and row dimension of test matrices to N.", TYPE_INT, &ni},
                              {'b', "-b B", "Set the mxaimum number of digits of integers to generate.", TYPE_INT, &bits},
                              {'i', "-i I", "Set the number of times to do the random unit tests.", TYPE_INT, &niter},
                              END_OF_ARGUMENTS};
    parseArguments(argc, argv, args);

#ifdef __LINBOX_HAVE_MPI
    MPI_Bcast(&ni, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&niter, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    srand(time(NULL));

    nj = ni;

    Givaro::ZRing<Integer> ZZ;
    DenseMatrix<Givaro::ZRing<Integer>> A(ZZ, ni, nj);

    using DenseVector = BlasVector<Givaro::ZRing<Integer>>;
    DenseVector X(ZZ, A.rowdim()), X2(ZZ, A.coldim()), B(ZZ, A.coldim());

    for (long j = 0; j < (long)niter; j++) {
        if (communicator.master()) {
            genData(ZZ, A, bits);
            genData(ZZ, B, bits);
        }

#ifdef __LINBOX_HAVE_MPI
        communicator.bcast(A, 0);
        communicator.bcast(B, 0);
#endif

        if (!test_set(X2, A, B, &communicator)) break;
    }

    return 0;
}
