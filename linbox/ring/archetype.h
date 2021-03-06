/* linbox/ring/archetype.h
 * Copyright(C) LinBox
 * Written by J-G Dumas <Jean-Guillaume.Dumas@imag.fr>,
 *            Clement Pernet <Clement.Pernet@imag.fr>
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
 *
 */

/*!@file ring/archetype.h
 * @ingroup ring
 * @brief Specification and archetypic instance for the ring interface.
 * @see \ref Archetypes
 */



#ifndef __LINBOX_ring_archetype_H
#define __LINBOX_ring_archetype_H

#include <iostream>
#include "linbox/field/archetype.h"
#include "linbox/ring/ring-interface.h"
#include "linbox/ring/abstract.h"
#include "linbox/ring/envelope.h"
#include "linbox/element/archetype.h"
#include "linbox/element/abstract.h"
#include "linbox/element/envelope.h"
#include "linbox/randiter/abstract.h"
#include "linbox/randiter/envelope.h"
#include "linbox/randiter/archetype.h"
#include "linbox/integer.h"
#include "linbox/linbox-config.h"

#include "linbox/util/error.h"

namespace LinBox
{
	// Forward declarations
	class RandIterArchetype;

	/**
	 * \brief specification and archetypic instance for the ring interface
	 \ingroup ring
	 *
	 * The \ref RingArchetype and its encapsulated
	 * element class contain pointers to the \ref RingAbstract
	 * and its encapsulated ring element, respectively.
	 * \ref RingAbstract then uses virtual member functions to
	 * define operations on its encapsulated ring element.  This ring
	 * element has no knowledge of the ring properties being used on it
	 * which means the ring object must supply these operations.
	 *
	 * It does not contain elements zero and one because they can be created
	 * whenever necessary, although it might be beneficial from an efficiency
	 * stand point to include them.  However, because of archetype use three,
	 * the elements themselves cannot be contained, but rather pointers to them.
	 */
	class RingArchetype : public virtual FieldArchetype {
	public:

		/** @name Common Object Interface for a LinBox Ring.
		 * These methods are required of all \ref LinBox rings.
		 */
		//@{

		/// element type.
		/* 		typedef ElementArchetype Element; */
		typedef FieldArchetype::Element Element;
		/// Random iterator generator type.
		/* 		typedef RandIterArchetype RandIter; */
		typedef FieldArchetype::RandIter RandIter;
		/// @name Object Management
		//@{

		/** Copy constructor.
		 *
		 * Constructs RingArchetype object by copying the
		 * ring.  This is required to allow ring objects to
		 * be passed by value into functions.
		 *
		 * In this implementation, this means copying the
		 * ring to which \c F._ring_ptr points, the
		 * element to which \c F._elem_ptr points, and the
		 * random element generator to which
		 * \c F._randIter_ptr points.
		 *
		 * @param F \ref RingArchetype object.
		 */
		RingArchetype (const RingArchetype &F) :
		       	FieldArchetype ( F )
		{ }

		/** \brief Invertibility test.
		 * Test if ring element is invertible.
		 * This function assumes the ring element has already been
		 * constructed and initialized.
		 * In this implementation, this means the \c
		 * _elem_ptr of x exists and does not point to
		 * null.
		 *
		 * @return boolean true if equals zero, false if not.
		 * @param  x ring element.
		 */
		bool isUnit (const Element &x) const
		{ return _ring_ptr->isUnit (*x._elem_ptr); }

		/** Divisibility of zero test.
		 * Test if ring element is a zero divisor.
		 * This function assumes the ring element has already been
		 * constructed and initialized.
		 *
		 * In this implementation, this means the \c
		 * _elem_ptr of x exists and does not point to
		 * null.
		 *
		 * @return boolean true if divides zero, false if not.
		 * @param  x ring element.
		 */
		bool isZeroDivisor (const Element &x) const
		{ return _ring_ptr->isZeroDivisor (*x._elem_ptr); }


		/** Constructor.
		 * Constructs ring from pointer to \ref RingAbstract and its
		 * encapsulated element and random element generator.
		 * Not part of the interface.
		 * Creates new copies of ring, element, and random iterator generator
		 * objects in dynamic memory.
		 * @param  ring_ptr pointer to \ref RingAbstract.
		 * @param  elem_ptr  pointer to \ref ElementAbstract, which is the
		 *                    encapsulated element of \ref RingAbstract.
		 * @param  randIter_ptr  pointer to \ref RandIterAbstract, which is the
		 *                        encapsulated random iterator generator
		 *                        of \ref RingAbstract.
		 */
		RingArchetype (RingAbstract    *ring_ptr,
			       ElementAbstract  *elem_ptr,
			       RandIterAbstract *randIter_ptr = 0) :
			FieldArchetype( static_cast<FieldAbstract*>(ring_ptr->clone()),
					elem_ptr, randIter_ptr ),
			_ring_ptr (dynamic_cast<RingAbstract*>(ring_ptr->clone ()))
		{ }


		/** Constructor.
		 * Constructs ring from ANYTHING matching the interface
		 * using the enveloppe as a \ref RingAbstract and its
		 * encapsulated element and random element generator if needed.
		 * @param f
		 */
		template<class Ring_qcq>
		RingArchetype (Ring_qcq *f)
		{ Ring_constructor (f, f); }

		//@} Implementation-Specific Methods

	private:

		friend class ElementArchetype;
		friend class RandIterArchetype;

		/** Pointer to RingAbstract object.
		 * Not part of the interface.
		 * Included to allow for archetype use three.
		 */
		mutable RingAbstract *_ring_ptr;


		/** Template method for constructing archetype from a derived class of
		 * RingAbstract.
		 * This class is needed to help the constructor differentiate between
		 * classes derived from RingAbstract and classes that aren't.
		 * Should be called with the same argument to both parameters?
		 * @param	trait	pointer to RingAbstract or class derived from it
		 * @param	ring_ptr	pointer to class derived from RingAbstract
		 */
		template<class Ring_qcq>
		void Ring_constructor (RingAbstract *trait,
				       Ring_qcq     *ring_ptr)
		{
			constructor( static_cast<FieldAbstract*>(trait), ring_ptr);
			_ring_ptr    = dynamic_cast<RingAbstract*>(ring_ptr->clone ());

		}

		/** Template method for constructing archetype from a class not derived
		 * from RingAbstract.
		 * This class is needed to help the constructor differentiate between
		 * classes derived from RingAbstract and classes that aren't.
		 * Should be called with the same argument to both parameters?
		 * @param	trait	pointer to class not derived from RingAbstract
		 * @param	ring_ptr	pointer to class not derived from RingAbstract
		 */
		template<class Ring_qcq>
		void Ring_constructor (void      *trait,
				       Ring_qcq *ring_ptr)
		{
			RingEnvelope< Ring_qcq > EnvF (*ring_ptr);
			Ring_constructor (static_cast<RingAbstract*> (&EnvF), &EnvF) ;
		}

	}; // class RingArchetype

	}  // namespace LinBox


#endif // __LINBOX_ring_archetype_H

// Local Variables:
// mode: C++
// tab-width: 4
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
// vim:sts=4:sw=4:ts=4:et:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
