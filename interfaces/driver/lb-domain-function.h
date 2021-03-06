/* lb-domain-function.h
 * Copyright (C) 2005 Pascal Giorgi
 *
 * Written by Pascal Giorgi <pgiorgi@uwaterloo.ca>
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

#ifndef __LINBOX_lb_domain_function_H
#define __LINBOX_lb_domain_function_H


#include <lb-domain-abstract.h>
#include <lb-domain-functor.h>


/*********************************************************
 * API to launch a generic function over a linbox domain *
 *********************************************************/
extern DomainTable domain_hashtable;

class DomainFunction {
public:

	// call a functor over a domain from the hashtable, result is given through 1st parameter
	template <class Functor, class Result>
	static void call (Result &res, const std::pair<const DomainKey, DomainAbstract*> &domain, const Functor &functor){
		ApplyDomainFunctor<Functor, Result> Ap(res, functor);
		(domain.second)->Accept(Ap);
	}

	// call a functor over a domain from the hashtable, no result
	template <class Functor>
	static void call (const std::pair<const DomainKey, DomainAbstract*>& k, const Functor& f){
		void *dumbresult;
		call(dumbresult, k, f);
	}

	// call a functor over a domain from its key, result is given through 1st parameter
	template <class Functor, class Result>
	static void call (Result &res, const DomainKey &key, const Functor &functor){
		DomainTable::iterator it = domain_hashtable.find(key);
		if (it != domain_hashtable.end())
			DomainFunction::call(res, *it, functor);
		else
			throw lb_runtime_error("LinBox ERROR: use of a non allocated domain\n");// throw an exception
	}

	// call a functor over a domain from its key, no result
	template <class Functor>
	static void call (const DomainKey &k, const Functor &f){
		void *dumbresult;
		call(dumbresult, k, f);
	}


};

#endif

// Local Variables:
// mode: C++
// tab-width: 4
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
// vim:sts=4:sw=4:ts=4:et:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
