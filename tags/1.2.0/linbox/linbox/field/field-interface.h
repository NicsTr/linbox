/* -*- mode: C++; tab-width: 8; indent-tabs-mode: t; c-basic-offset: 8 -*- */
// vim:sts=8:sw=8:ts=8:noet:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
/* linbox/field/field-interface.h
 * Copyright (C) 2002 David Saunders
 *
 * For licensing information see COPYING
 */

#ifndef __LINBOX_field_interface_H
#define __LINBOX_field_interface_H

namespace LinBox
{
	// LinBox Field Interface
	/*
	 * The LinBox {@link Fields field} common object {@link Interfaces interface}.
	 * The field interface includes the following public members:
	 *
	 * Types: \c Element and \c RandIter.
	 *
	 * Object management member functions:
	 *   null constructor, copy constructor, destructor, assignment operator,
	 *   \c convert(), \c init(), \c assign(), \c characteristic(), \c cardinality().
	 *
	 * Predicates on field elements:
	 *   \c areEqual(), \c isZero(), \c isOne().
	 *
	 * Basic arithmetic functions:
	 *   \c axpy(), \c add(), \c neg(), \c sub(), \c mul(), \c inv(),\c  div().
	 *
	 * Inplace arithmetic functions:
	 *   \c axpyin(), \c addin(), \c negin(), \c subin(), \c mulin(), \c invin(), \c divin().
	 *
	 * I/O functions:
	 *   \c read() and \c write() for I/O of the field itself and for I/O of its elements.
	 *
	 * The field archetype class is is the reference instantiation of this
	 * interface and contains the generic specifications of the member functions.
	 * Documentation in other field classes is more limited. It serves primarily to explain special properties
	 * specific to the class of the interface member functions and to explain any constructors
	 * or other functionality unique to the class.
	 *
	 *  @see Interfaces
	 */
	/**
	 * \brief This field base class exists solely to aid documentation organization.


	 *  For the general field member function documentation consult the
	 *  @link FieldArchetype FieldArchetype@endlink. For specific properties of individual representations consult the specific field classes.
	 \ingroup field
	 */
	class FieldInterface {
#if 0
	public:
		// this just demo's that some declarations could be here.
		typedef ElementArchetype Element;
		virtual Element& mul(Element& c, const Element& a, const Element& b) const = 0;
#endif
	};// empty class so doc++ makes a nice hierarchy.

} // namespace LinBox

#endif // __LINBOX_field_interface_H
