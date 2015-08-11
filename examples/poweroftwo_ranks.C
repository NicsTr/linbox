/* examples/poweroftwo_ranks.C
 *
 * Copyright (C) 2012 LinBox
 * Written by J-G Dumas
 * Time-stamp: <27 Mar 14 10:19:58 Jean-Guillaume.Dumas@imag.fr>
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 * ========LICENCE========
 */

/** \file examples/poweroftwo_ranks.C
 * @example  examples/poweroftwo_ranks.C
  \brief Ranks of sparse matrix modulo 2^k
  \ingroup examples
  */
#include <linbox/linbox-config.h>

#include <iostream>

#include <linbox/field/givaro.h>
#include <linbox/matrix/sparse-matrix.h>
#include <linbox/algorithms/smith-form-sparseelim-poweroftwo.h>

using namespace LinBox;
using namespace std;

template<class Int_type, class Ring_type = Givaro::ZRing<Int_type> >
void runpoweroftworank(ifstream& input, const size_t exponent) {
    typedef std::vector<std::pair<size_t,Int_type> > Smith_t;
    typedef Ring_type Ring; // signed ?
    Smith_t local;
    Ring R;
    LinBox::MatrixStream<Ring> ms( R, input );
    LinBox::SparseMatrix<Ring, LinBox::SparseMatrixFormat::SparseSeq > A (ms);

    input.close();
    LinBox::PowerGaussDomainPowerOfTwo< Int_type > PGD;
            
    cout << "B is " << A.rowdim() << " by " << A.coldim() << endl;
//     R.write(std::cerr << "Last entry: ", A.getEntry(A.rowdim()-1,A.coldim()-1)) << std::endl;

    LinBox::Timer tim; 
    tim.clear(); tim.start();
    PGD(local, A, exponent);
    tim.stop();

    R.write(std::cout << "Local Smith Form ") << " : " << std::endl << '(';
    for (auto  p = local.begin(); p != local.end(); ++p)
        std::cout << '[' << p->second << ',' << p->first << "] ";
    cout << ')' << endl;
       
    std::cerr << tim << std::endl;
}

int main (int argc, char **argv) {
    commentator().setMaxDetailLevel (-1);
    commentator().setMaxDepth (-1);
    commentator().setReportStream (std::cerr);

    if (argc < 3 || argc > 3)
    {	cerr << "Usage: rank <matrix-file-in-supported-format> <power of two exponent>]" << endl; return -1; }

    ifstream input (argv[1]);
    if (!input) { cerr << "Error opening matrix file: " << argv[1] << endl; return -1; }

    // long unsigned int r;

    if (argc == 3) {
        LinBox::Timer tim;
        size_t exponent = atoi(argv[2]);
        if (exponent > 63) {
            runpoweroftworank<Givaro::Integer, LinBox::PID_integer>(input, exponent);

        } else {
            runpoweroftworank<uint64_t, Givaro::ZRing<int64_t> >(input, exponent);
            
        }



        std::cerr << tim << std::endl;
    }

    return 0;
}

// Local Variables:
// mode: C++
// tab-width: 8
// indent-tabs-mode: nil
// c-basic-offset: 8
// End:
// vim:sts=8:sw=8:ts=8:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s