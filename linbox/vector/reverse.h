/* linbox/vector/reverse.h
 * Copyright (C) 2002 Bradford Hovinen
 *
 * ------------------------------------
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
 *.
 */

#ifndef __LINBOX_reverse_H
#define __LINBOX_reverse_H

#include "vector-traits.h"

#include <iterator>
#include <vector>
#include <stdexcept>

namespace LinBox
{
	/** Reverse vector class
	 * This class wraps an existing vector type and reverses its
	 * direction. It is used as an adaptor to allow VectorDomain dot
	 * products to be used for the Massy implementation.
	 \ingroup vector
	 */
	template <class Vector>
	class ReverseVector {
	    public:
		typedef typename Vector::value_type             value_type;
		typedef typename Vector::size_type              size_type;
		typedef typename Vector::difference_type        difference_type;
		typedef typename Vector::pointer                pointer;
		typedef typename Vector::reference              reference;
		typedef typename Vector::const_reference        const_reference;
		typedef typename Vector::reverse_iterator       iterator;
		typedef typename Vector::const_reverse_iterator const_iterator;
		typedef typename Vector::iterator               reverse_iterator;
		typedef typename Vector::const_iterator         const_reverse_iterator;

		ReverseVector (Vector& v) :
		       	_v (v)
		{}

		// Copy constructor
		ReverseVector (const ReverseVector<Vector> &v) :
		       	_v (v._v)
		{}

		~ReverseVector () {}

		// Iterators

		inline iterator               begin  (void)       { return _v.rbegin (); }
		inline const_iterator         begin  (void) const { return _v.rbegin (); }
		inline iterator               end    (void)       { return _v.rend (); }
		inline const_iterator         end    (void) const { return _v.rend (); }

		inline reverse_iterator       rbegin (void)       { return _v.begin (); }
		inline const_reverse_iterator rbegin (void) const { return _v.begin (); }
		inline reverse_iterator       rend   (void)       { return _v.end (); }
		inline const_reverse_iterator rend   (void) const { return _v.end (); }

		// Element access

		inline reference       operator[] (size_type n)       { return ((begin ())[(ptrdiff_t)n]); }
		inline const_reference operator[] (size_type n) const { return ((begin ())[(ptrdiff_t)n]); }

		// the method "at" does appear to be implemented
		// in the gnu implementation of the STL
		reference at (size_type n)  // validity is relative to valid _begin, _end
		{
			iterator p = begin () + n;
			if (begin () <= p && p < end ())
				return *p;
			else
				throw std::out_of_range("out of range"); //out of range error message.
		}

		const_reference at(size_type n) const
		{
			const_iterator p = begin () + n;
			if (begin () <= p && p < end ())
				return *p;
			else
				throw std::out_of_range("out of range"); //out of range error message
		}

		inline reference       front (void)       { return *begin (); }
		inline const_reference front (void) const { return *begin (); }
		inline reference       back  (void)       { return *(end () - 1); }
		inline const_reference back  (void) const { return *(end () - 1); }

		template<class Container>
		/** assign the elements of Container one by one to *this.
		 *  Container must be at least as long as this.
		 */
		ReverseVector &operator= (const Container& x)
		{
			typename Container::const_iterator q = x.begin ();

			for (iterator p = begin (); p != end (); ++p, ++q)
				*p = *q;

			return *this;
		}

//		template <class In> void assign(In first, In last);
//		void assign(size_type n, const T& val);

		// Stack operations:
		// 	not implemented because they invalidate iterators

		// List operations:
		// 	not implemented because they invalidate iterators

		// Capacity
		// 	resize, reserve: not implemented because they
		// 		invalidate iterators

		inline size_type size      (void) const { return _v.size  (); }
		inline bool      empty     (void) const { return _v.empty (); }
		inline size_type max_size  (void) const { return _v.size  (); }

	    protected:

		Vector &_v;

	}; // template <class Vector> class ReverseVector

	// Vector traits for ReverseVector wrapper
	template <class Vector>
	struct VectorTraits<ReverseVector<Vector> >
	{
		typedef typename VectorTraits<Vector>::VectorCategory VectorCategory;
	};

} // namespace LinBox
#endif //__LINBOX_reverse_H

// Local Variables:
// mode: C++
// tab-width: 4
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
// vim:sts=4:sw=4:ts=4:et:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
