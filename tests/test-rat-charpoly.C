/* tests/test-solve.C
 * Copyright (C) 2001, 2002 Bradford Hovinen
 *
[12~ * Written by Bradford Hovinen <hovinen@cis.udel.edu>
 *
 * --------------------------------------------------------
 *
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 * ========LICENCE========
 *
 */


/*! @file  tests/test-solve.C
 * @ingroup tests
 * @brief  no doc
 * @test no doc.
 */



#include "linbox/linbox-config.h"

#include <iostream>
#include <fstream>

#include <cstdio>

#include "linbox/util/commentator.h"
#include "linbox/matrix/sparse-matrix.h"
#include "linbox/solutions/charpoly.h"

#include "test-common.h"

using namespace LinBox;

/* Test 1: Charpoly of a diagonal matrix
 *
 * Constructs a random diagonal rational matrix D
 * Computes its characteristic polynomial
 * Checks if c[0] (determinant) and c[n-1] (trace) agree
 *
 * Return true on success and false on failure
 */

static bool testDiagRatCharpoly (size_t n, int iterations)
{
	commentator().start ("Testing rational charpoly of diagonal matrix ", "testNonsingularRatIntSolve", (unsigned int)iterations);

	bool ret = true;
	int i;
	size_t j;

	GMPRationalField Q;
	SparseMatrix<GMPRationalField > A(Q,n,n);
	BlasMatrix <GMPRationalField > B(Q,n,n);
	BlasVector<GMPRationalField> c(Q);

	for (i=0; i < iterations; i++) {
        	GMPRationalField::Element c0,cn;
	        Q.init(c0,1,1);
	        Q.init(cn,0,1);


		commentator().startIteration ((unsigned int)i);

		for (j=0; j < n; ++j) {
			integer tmp_n, tmp_d;
			GMPRationalField::Element tmp, abstmp;
			tmp_n = (integer) rand() % (5*(i +1)) + 1;
			tmp_d = (integer) rand() % (5*(i +1)) + 1;
			if ( ( i%2) && (j % 2)) integer::negin(tmp_n);

			Q.init(tmp, tmp_n,tmp_d);

			A.setEntry(j,j,tmp);
			B.setEntry(j,j,tmp);


			Q.mulin(c0, tmp);
			Q.addin(cn, tmp);
		}
		if (n%2==0) Q.negin(cn);

		// ostream &report = commentator().report (Commentator::LEVEL_IMPORTANT, INTERNAL_DESCRIPTION);
		charpoly (c, A);
		if ((!Q.areEqual(c[0] , c0)) || (!Q.areEqual(c[n-1] , cn) ) ) {
			commentator().report() << "ERROR: Sparse charpoly failed" << endl;
			ret = false;
		}
		c.clear();

		charpoly (c, B);
		if ((!Q.areEqual(c[0] , c0)) || (!Q.areEqual(c[n-1] , cn) ) ) {
                        commentator().report() << "ERROR: Dense charpoly failed" << endl;
                        ret = false;
                }

		commentator().stop ("done");
		commentator().progress ();
	}

	commentator().stop (MSG_STATUS (ret), (const char *) 0, "testNonsingularRatIntSolve");

	return ret;
}

int main (int argc, char **argv)
{
	bool pass = true;

	static size_t n = 10;
	static int iterations = 1;

	static Argument args[] = {
		{ 'n', "-n N", "Set column dimension of test matrices to N.", TYPE_INT,     &n },
		{ 'i', "-i I", "Perform each test for I iterations.", TYPE_INT,     &iterations },
		END_OF_ARGUMENTS
	};
	parseArguments (argc, argv, args);

	commentator().start("Rational solve test suite", "solve");

	commentator().getMessageClass (INTERNAL_DESCRIPTION).setMaxDepth (10);
	commentator().getMessageClass (INTERNAL_DESCRIPTION).setMaxDetailLevel (Commentator::LEVEL_NORMAL);
	commentator().getMessageClass (TIMING_MEASURE).setMaxDepth (10);
	commentator().getMessageClass (TIMING_MEASURE).setMaxDetailLevel (Commentator::LEVEL_NORMAL);
	commentator().getMessageClass (PROGRESS_REPORT).setMaxDepth (5);
	//commentator().getMessageClass (BRIEF_REPORT).setMaxDepth (4);

    if ( ! testDiagRatCharpoly(n,iterations) ) pass = false;

	commentator().stop("solve test suite");
    //std::cout << (pass ? "passed" : "FAILED" ) << std::endl;

	return pass ? 0 : -1;
}

// Local Variables:
// mode: C++
// tab-width: 4
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
// vim:sts=4:sw=4:ts=4:et:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
