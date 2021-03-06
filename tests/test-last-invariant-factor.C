/* Copyright (C) LinBox
 *
 *  Author: Zhendong Wan
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 * ========LICENCE========
 */



/*! @file  tests/test-last-invariant-factor.C
 * @ingroup tests
 * @brief  no doc
 * @test NO DOC
 */



#include <linbox/linbox-config.h>
#include "givaro/zring.h"
#include "linbox/randiter/random-prime.h"
#include "linbox/ring/modular.h"
#include "linbox/algorithms/matrix-rank.h"
#include "linbox/algorithms/last-invariant-factor.h"
#include "linbox/blackbox/scompose.h"
#include "linbox/blackbox/random-matrix.h"
#include "linbox/algorithms/rational-solver.h"
#include <time.h>

#include "linbox/util/commentator.h"
#include "linbox/vector/stream.h"
#include "test-common.h"

using namespace LinBox;

template <class Ring, class LIF, class Vector>
bool testRandom(const Ring& R,
		const LIF& lif,
		LinBox::VectorStream<Vector>& stream1)
{

	std::ostringstream str;

	str << "Testing last invariant factor:";

        commentator().start (str.str ().c_str (), "testRandom", stream1.m ());

        bool ret = true;

        VectorDomain<Ring> VD (R);

	Vector d(R);

	typename Ring::Element x;

	VectorWrapper::ensureDim (d, stream1.n ());

	int n = int(d. size());

	 while (stream1) {

		 commentator().startIteration ((unsigned)stream1.j ());

		 std::ostream &report = commentator().report (Commentator::LEVEL_IMPORTANT, INTERNAL_DESCRIPTION);

                bool iter_passed = true;

		stream1.next (d);

		report << "Input vector:  ";
		VD.write (report, d);
                report << endl;

		BlasMatrix<Ring> D(R, n, n), L(R, n, n), U(R, n, n), A(R,n,n);

		int i, j;

		for(i = 0; i < n; ++i) {
			R. assign (D[(size_t)i][(size_t)i], d[(size_t)i]);
			R. assign (L[(size_t)i][(size_t)i], R.one);
			R. assign (U[(size_t)i][(size_t)i], R.one);}

		for (i = 0; i < n; ++ i)

			for (j = 0; j < i; ++ j) {

				R.init(L[(size_t)i][(size_t)j], int64_t(rand() % 10));

				R.init(U[(size_t)j][(size_t)i], int64_t(rand() % 10));
			}


		BlasVector<Ring> tmp1(R,(size_t)n), tmp2(R,(size_t)n), e(R,(size_t)n);

		typename BlasMatrix<Ring>::ColIterator col_p;

		i = 0;
		for (col_p = A.colBegin();
		     col_p != A.colEnd(); ++ col_p, ++ i) {

			R.assign(e[(size_t)i],R.one);
			U.apply(tmp1, e);
			D.apply(tmp2, tmp1);
			// LinBox::BlasSubvector<BlasVector<Ring> > col_p_v(R,*col_p);
			// L.apply(col_p_v, tmp2);
			L.apply(*col_p, tmp2);
			R.assign(e[(size_t)i],R.zero);
		}



		lif. lastInvariantFactor (x, A);


		report << "Computed last invariant factor: \n";

		R. write (report, x);

		report << '\n';


		typename BlasVector<Ring>::iterator p1;

		typename Ring::Element l;

		R. assign(l , R.one);

		for (p1 = d.begin(); p1 != d.end(); ++ p1)

			R. lcmin (l, *p1);



		report << "Expected last invariant factor: \n";

		R. write (report, l);

		report << '\n';

		if (!R. areEqual (l, x))

			ret = iter_passed = false;

                if (!iter_passed)

                        commentator().report (Commentator::LEVEL_IMPORTANT, INTERNAL_ERROR)
				<< "ERROR: Computed last invariant factor is incorrect" << endl;



                commentator().stop ("done");

                commentator().progress ();

	 }

	 //stream1.reset ();

	  commentator().stop (MSG_STATUS (ret), (const char *) 0, "testRandom");

	  return ret;

}

int main(int argc, char** argv)
{


        bool pass = true;

        static size_t n = 10;

	static unsigned int iterations = 1;

        static Argument args[] = {
                { 'n', "-n N", "Set order of test matrices to N.", TYPE_INT,     &n },
                { 'i', "-i I", "Perform each test for I iterations.", TYPE_INT,     &iterations },
		END_OF_ARGUMENTS

        };

	parseArguments (argc, argv, args);

        typedef Givaro::ZRing<Integer>      Ring;

        Ring R; Ring::RandIter gen(R);

	commentator().start("Last invariant factor test suite", "LIF");

        commentator().getMessageClass (INTERNAL_DESCRIPTION).setMaxDepth (5);

        RandomDenseStream<Ring> s1 (R, gen, n, iterations);

	typedef RationalSolver<Ring, Givaro::Modular<int32_t>, PrimeIterator<IteratorCategories::HeuristicTag> > Solver;
        // typedef RationalSolver<Ring, Givaro::Modular<double>, LinBox::PrimeIterator<IteratorCategories::HeuristicTag> > Solver;

	typedef LastInvariantFactor<Ring, Solver> LIF;

	LIF lif;

	lif.  setThreshold (30);

	if (!testRandom(R, lif, s1)) pass = false;

	commentator().stop("Last invariant factor test suite");
        return pass ? 0 : -1;
}

// Local Variables:
// mode: C++
// tab-width: 4
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
// vim:sts=4:sw=4:ts=4:et:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
