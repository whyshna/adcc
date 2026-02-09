#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
from math import sqrt

from adcc import block as b
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

from .util import check_doubles_amplitudes, check_singles_amplitudes


def tdm_adc0(mp, amplitude, intermediates):
    # C is either c(ore) or o(ccupied)
    C = b.c if mp.has_core_occupied_space else b.o
    check_singles_amplitudes([C, b.v], amplitude)
    u1 = amplitude.ph

    # Transition density matrix for (CVS-)ADC(0)
    dm = OneParticleOperator(mp, is_symmetric=False)
    dm[b.v + C] = u1.transpose()
    return dm


def tdm_adc1(mp, amplitude, intermediates):
    dm = tdm_adc0(mp, amplitude, intermediates)  # Get ADC(0) result
    # adc1_dp0_ov
    dm.ov = -einsum("ijab,jb->ia", mp.t2(b.oovv), amplitude.ph)
    return dm


def tdm_cvs_adc2(mp, amplitude, intermediates):
    # Get CVS-ADC(1) result (same as CVS-ADC(0))
    dm = tdm_adc0(mp, amplitude, intermediates)
    check_doubles_amplitudes([b.o, b.c, b.v, b.v], amplitude)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = intermediates.cvs_p0

    # Compute CVS-ADC(2) tdm
    dm.oc = (  # cvs_adc2_dp0_oc
        - einsum("ja,Ia->jI", p0.ov, u1)
        + (1 / sqrt(2)) * einsum("kIab,jkab->jI", u2, t2)
    )

    # cvs_adc2_dp0_vc
    dm.vc -= 0.5 * einsum("ab,Ib->aI", p0.vv, u1)
    return dm


def tdm_adc2(mp, amplitude, intermediates):
    dm = tdm_adc1(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    # Compute ADC(2) tdm
    dm.oo = (  # adc2_dp0_oo
        - einsum("ia,ja->ij", p0.ov, u1)
        - einsum("ikab,jkab->ji", u2, t2)
    )
    dm.vv = (  # adc2_dp0_vv
        + einsum("ia,ib->ab", u1, p0.ov)
        + einsum("ijac,ijbc->ab", u2, t2)
    )
    dm.ov -= einsum("ijab,jb->ia", td2, u1)  # adc2_dp0_ov
    dm.vo += 0.5 * (  # adc2_dp0_vo
        + einsum("ijab,jkbc,kc->ai", t2, t2, u1)
        - einsum("ab,ib->ai", p0.vv, u1)
        + einsum("ja,ij->ai", u1, p0.oo)
    )
    return dm

def tdm_adc3(mp, intermediates, amplitude):
    dm  = tdm_adc2(mp, amplitude, intermediates)
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    
    ul1, ul2 = amplitude.ph, amplitude.pphh  #adc amplitudes

    t1_2 = mp.mp2.diffdm.ov
    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)
    

    p0_3_ov = mp.mp3_dm_correction.ov
    p0_3_oo = mp.mp3_dm_correction.oo
    p0_3_vv = mp.mp3_dm_correction.vv

    p0_2_oo = mp.mp2_dm_correction.oo
    p0_2_vv = mp.mp2_dm_correction.vv

    #ADC(3) 1tdm
    dm.oo = (
        - 1 * einsum("ja,ia->ij", ul1, t1_2)  
	- 1 * einsum("ja,ia->ij", ul1, p0_3_ov)  
	- 1 * einsum("jkab,ikab->ij", ul2, t2_1)  
	- 1 * einsum("jkab,ikab->ij", ul2, t2_2)  
	+ 1 * einsum("ia,ja->ij", einsum("kb,ikab->ia", ul1, t2_1), t1_2)  
	+ 0.5 * einsum("ikbc,jkbc->ij", einsum("la,iklabc->ikbc", ul1, t3_2), t2_1)  
)

    dm.vv = (
        + 1 * einsum("ia,ib->ab", ul1, t1_2)  
	+ 1 * einsum("ia,ib->ab", ul1, p0_3_ov)  
	+ 1 * einsum("ijac,ijbc->ab", ul2, t2_1)  
	+ 1 * einsum("ijac,ijbc->ab", ul2, t2_2) 
	+ 0.5 * einsum("ikbc,ikac->ab", einsum("jd,ijkbcd->ikbc", ul1, t3_2), t2_1)  
	- 1 * einsum("ib,ia->ab", einsum("jc,ijbc->ib", ul1, t2_1), t1_2) 

    )

    dm.ov = (
        - 1 * einsum("jb,ijab->ia", ul1, t2_1)  
	- 1 * einsum("jb,ijab->ia", ul1, t2_2) 
	- 1 * einsum("jb,ijab->ia", ul1, t2_3) 
	- 0.5 * einsum("jkbc,ijkabc->ia", ul2, t3_2) 
	+ 1 * einsum("ka,ik->ia", einsum("jb,jkab->ka", ul1, t2_1), p0_2_oo)  
	+ 0.5 * einsum("jc,ijac->ia", einsum("jb,bc->jc", ul1, p0_2_vv), t2_1)  
	- 1 * einsum("ic,ac->ia", einsum("jb,ijbc->ic", ul1, t2_1), p0_2_vv) 
	- 0.5 * einsum("kb,ikab->ia", einsum("jb,jk->kb", ul1, p0_2_oo), t2_1)  
	+ 1 * einsum("ijkc,jkac->ia", einsum("ijld,klcd->ijkc", einsum("jb,ilbd->ijld", ul1, t2_1), t2_1), t2_1) 
	+ 0.5 * einsum("kd,ikad->ia", einsum("lc,klcd->kd", einsum("jb,jlbc->lc", ul1, t2_1), t2_1), t2_1)  
	- 0.25 * einsum("ijkl,jkla->ia", einsum("ijcd,klcd->ijkl", t2_1, t2_1), einsum("jb,klab->jkla", ul1, t2_1)) 

	)

    dm.vo = (
        + 0.5 * einsum("ja,ij->ai", ul1, p0_2_oo) 
	+ 0.5 * einsum("ja,ij->ai", ul1, p0_3_oo)  
	- 0.5 * einsum("ib,ab->ai", ul1, p0_2_vv) 
	- 0.5 * einsum("ib,ab->ai", ul1, p0_3_vv) 
	+ 0.5 * einsum("kc,ikac->ai", einsum("jb,jkbc->kc", ul1, t2_2), t2_1)
	+ 0.5 * einsum("jb,ijab->ai", einsum("kc,jkbc->jb", ul1, t2_1), t2_1)  
	+ 0.5 * einsum("jb,ijab->ai", einsum("kc,jkbc->jb", ul1, t2_1), t2_2)  
	+ 1 * einsum("ia->ai", ul1)  
	)
    return dm

DISPATCH = {
    "adc0": tdm_adc0,
    "adc1": tdm_adc1,
    "adc2": tdm_adc2,
    "adc2x": tdm_adc2,
    "cvs-adc0": tdm_adc0,
    "cvs-adc1": tdm_adc0,  # No extra contribs for CVS-ADC(1)
    "cvs-adc2": tdm_cvs_adc2,
    "cvs-adc2x": tdm_cvs_adc2,
    "adc3": tdm_adc3,
}


def transition_dm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle transition density matrix from ground to excited
    state in the MO basis.

    Parameters
    ----------
    method : str, AdcMethod
        The method to use for the computation (e.g. "adc2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude : AmplitudeVector
        The amplitude vector
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude, AmplitudeVector):
        raise TypeError("amplitude should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("transition_dm is not implemented "
                                  f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
