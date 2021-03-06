/* lb-det.h
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

#ifndef __LINBOX_lb_det_H
#define __LINBOX_lb_det_H

#include <lb-element-collection.h>
#include <lb-blackbox-collection.h>


/*******************************************************
 * API for determinant computation                     *
 * determinant is returned through a given element key *
 *******************************************************/
void lb_determinant(const EltKey &Ekey, const BlackboxKey &Bkey, const char *method="hybrid");


/**************************************************
 * API for determinant computation                *
 * determinant is returned through an element key *
 **************************************************/
const EltKey& lb_determinant(const BlackboxKey &key, const char *method="hybrid");


/*************************************************************
 * API to print available method for determinant computation *
 *************************************************************/
const char* lb_determinant_methods();


#endif

// Local Variables:
// mode: C++
// tab-width: 4
// indent-tabs-mode: nil
// c-basic-offset: 4
// End:
// vim:sts=4:sw=4:ts=4:et:sr:cino=>s,f0,{0,g0,(0,\:0,t0,+0,=s
