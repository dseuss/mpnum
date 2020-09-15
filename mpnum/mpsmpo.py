# encoding: utf-8


r"""Matrix Product State (MPS) and Operator (MPO) functions

The :ref:`mpnum-introduction` also covers the definitions mentioned
below.

.. _mpsmpo-definitions:

Definitions
-----------

We consider a linear chain of :math:`n` sites with associated Hilbert
spaces \mathcal H_k = \C^{d_k}, :math:`d_k`, :math:`k \in [1..n] :=
\{1, 2, \ldots, n\}`. The set of linear operators :math:`\mathcal H_k
\to \mathcal H_k` is denoted by :math:`\mathcal B_k`. We write
:math:`\mathcal H = \mathcal H_1 \otimes \cdots \otimes \mathcal H_n`
and the same for :math:`\mathcal B`.

We use the following three representations:

* Matrix product state (MPS): Vector :math:`\lvert \psi \rangle \in
  \mathcal H`

* Matrix product operator (MPO): Operator :math:`M \in \mathcal B`

* Locally purified matrix product state (PMPS): Positive semidefinite
  operator :math:`\rho \in \mathcal B`

All objects are represented by :math:`n` local tensors.

Matrix product state (MPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Represent a vector :math:`\lvert \psi \rangle \in \mathcal H` as

.. math::

   \langle i_1 \ldots i_n \vert \psi \rangle
   = A^{(1)}_{i_1} \cdots A^{(n)}_{i_n},
   \quad A^{(k)}_{i_k} \in \mathbb C^{D_{k-1} \times D_k},
   \quad D_0 = 1 = D_n.

The :math:`k`-th local tensor is :math:`T_{l,i,r} =
(A^{(k)}_i)_{l,r}`.

The vector :math:`\lvert \psi \rangle` can be a quantum state, with
the density matrix given by :math:`\rho = \lvert \psi \rangle \langle
\psi \rvert \in \mathcal B`.  Reference: E.g. [Sch11]_.

Matrix product operator (MPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Represent an operator :math:`M \in \mathcal B` as

.. math::

  \langle i_1 \ldots i_n \vert M \vert j_1 \ldots j_n \rangle
  = A^{(1)}_{i_1 j_1} \cdots A^{(n)}_{i_n j_n},
  \quad A^{(k)}_{i_k j_k} \in \mathbb C^{D_{k-1} \times D_k},
   \quad D_0 = 1 = D_n.

The :math:`k`-th local tensor is :math:`T_{l,i,j,r} = (A^{(k)}_{i
j})_{l,r}`.

This representation can be used to represent a mixed quantum state
:math:`\rho = M`, but it is not limited to positive semidefinite
:math:`M`.  Reference: E.g. [Sch11]_.

Locally purified matrix product state (PMPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Represent a positive semidefinite operator :math:`\rho \in \mathcal B`
as follows: Let :math:`\mathcal H_k' = \mathbb C^{d'_k}` with suitable
:math:`d'_k` and :math:`\mathcal P = \mathcal H_1 \otimes \mathcal
H'_1 \otimes \cdots \otimes \mathcal H_n \otimes \mathcal H'_n`. Find
:math:`\vert \Phi \rangle \in \mathcal P` such that

.. math::

   \rho = \operatorname{tr}_{\mathcal H'_1, \ldots, \mathcal H'_n}
   (\lvert \Phi \rangle \langle \Phi \rvert)

and represent :math:`\lvert \Phi \rangle` as

.. math::

   \langle i_1 i'_1 \ldots i_n i'_n \vert \Phi \rangle
   = A^{(1)}_{i_1 i'_1} \cdots A^{(n)}_{i_n i'_n},
   \quad A^{(k)}_{i_k j_k} \in \mathbb C^{D_{k-1} \times D_k},
   \quad D_0 = 1 = D_n.

The :math:`k`-th local tensor is :math:`T_{l,i,i',r} = (A^{(k)}_{i
i'})_{l,r}`.

The ancillary dimensions :math:`d'_i` are not determined by the
:math:`d_i` but depend on the state. E.g. if :math:`\rho` is pure, one
can set all :math:`d_i = 1`.  Reference: E.g. [Cue13_].


.. todo:: Are derived classes MPO/MPS/PMPS of any help?

.. todo:: I am not sure the current definition of PMPS is the most elegant
          for our purposes...


References:

* .. _Cue13:

  [Cue13] De las Cuevas, G., Schuch, N., Pérez-García, D., and Cirac,
  J. I. (2013). “Purifications of multipartite states: limitations and
  constructive methods”. New J. Phys. 15(12), p. 123021. `DOI:
  10.1088/1367-2630/15/12/123021`_. `arXiv: 1308.1914`_.

  .. _`DOI: 10.1088/1367-2630/15/12/123021`:
     http://dx.doi.org/10.1088/1367-2630/15/12/123021

  .. _`arXiv: 1308.1914`: http://arxiv.org/abs/1308.1914


"""

from __future__ import absolute_import, division, print_function

import numpy as np

from six.moves import range

from . import mparray as mp
from .utils import local_to_global, matdot

from scipy.linalg import eigh

__all__ = ['mps_to_mpo', 'mps_to_pmps', 'pmps_dm_to_array',
           'pmps_reduction', 'mpo_to_pmps', 'pmps_to_mpo', 'pmps_to_mps',
           'reductions_mpo', 'reductions_mps_as_mpo',
           'reductions_mps_as_pmps', 'reductions_pmps', 'reductions']


def _check_reductions_args(nr_sites, width, startsites, stopsites):
    """Expand the arguments of :func:`reductions_mpo()` et al.

    The arguments are documented in :func:`reductions_mpo()`.

    """
    if stopsites is None:
        assert width is not None
        if startsites is None:
            startsites = range(nr_sites - width + 1)
        else:
            startsites = tuple(startsites)  # Allow iterables
        stopsites = (start + width for start in startsites)
    else:
        assert width is None
        assert startsites is not None
    return startsites, stopsites


def pmps_dm_to_array(pmps, global_=False):
    """Convert PMPS to full array representation of the density matrix

    The runtime of this method scales with D**3 instead of D**6 where
    D is the rank and D**6 is the scaling of using :func:`pmps_to_mpo`
    and :func:`to_array`. This is useful for obtaining reduced states
    of a PMPS on non-consecutive sites, as normalizing before using
    :func:`pmps_to_mpo` may not be sufficient to reduce the rank
    in that case.

    .. note:: The resulting array will have dimension-1 physical legs removed.

    """
    out = np.ones((1, 1, 1))
    # Axes: 0 phys, 1 upper rank, 2 lower rank
    for lt in pmps.lt:
        out = np.tensordot(out, lt, axes=(1, 0))
        # Axes: 0 phys, 1 lower rank, 2 phys, 3 anc, 4 upper rank
        out = np.tensordot(out, lt.conj(), axes=((1, 3), (0, 2)))
        # Axes: 0 phys, 1 phys, 2 upper rank, 3 phys, 4 lower rank
        out = np.rollaxis(out, 3, 2)
        # Axes: 0 phys, 1 phys, 2 phys, 3 upper bound, 4 lower rank
        out = out.reshape((-1, out.shape[3], out.shape[4]))
        # Axes: 0 phys, 1 upper rank, 2 lower rank
    out_shape = [dim for dim, _ in pmps.shape for rep in (1, 2) if dim > 1]
    out = out.reshape(out_shape)
    if global_:
        assert len(set(out_shape)) == 1
        out = local_to_global(out, sites=len(out_shape) // 2)
    return out


def pmps_reduction(pmps, support):
    """Convert a PMPS to a PMPS representation of a local reduced state

    :param support: Set of sites to keep

    :returns: Sites traced out at the beginning or end of the chain
        are removed using :func:`reductions_pmps` and a suitable
        normalization. Sites traced out in the middle of the chain are
        converted to sites with physical dimension 1 and larger
        ancilla dimension.

    """
    n_sites = len(pmps)
    assert len(support) > 0
    assert all(0 <= s < n_sites for s in support)
    start_at = min(support)
    stop_at = max(support) + 1
    red = next(reductions_pmps(pmps, startsites=[start_at], stopsites=[stop_at]))
    width = stop_at - start_at
    if len(support) == width:
        return red
    support = [pos - start_at for pos in support]
    return mp.MPArray(
        lt if pos in support else lt.reshape((lt.shape[0], 1, -1, lt.shape[-1]))
        for pos, lt in enumerate(red.lt)
    )


def reductions_mpo(mpa, width=None, startsites=None, stopsites=None):
    """Iterate over MPO partial traces of an MPO

    The support of the i-th result is :code:`range(startsites[i],
    stopsites[i])`.

    :param mpnum.mparray.MPArray mpa: An MPO

    :param startsites: Defaults to :code:`range(len(mpa) - width +
        1)`.

    :param stopsites: Defaults to :code:`[ start + width for start in
        startsites ]`. If specified, we require `startsites` to be
        given and `width` to be None.

    :param width: Number of sites in support of the results. Default
        `None`. Must be specified if one or both of `startsites` and
        `stopsites` are not given.

    :returns: Iterator over partial traces as MPO

    """
    startsites, stopsites = \
        _check_reductions_args(len(mpa), width, startsites, stopsites)

    if not (np.array(mpa.ndims) == 2).all():
        raise ValueError('Every site needs to have exactly two physical legs')
    rem_left = {0: np.array(1, ndmin=2)}
    rem_right = rem_left.copy()

    def get_remainder(rem_cache, num_sites, end):
        """Obtain the vectors resulting from tracing over
        the left or right end of a Matrix Product Operator.

        :param rem_cache: Save remainder terms with smaller num_sites here
        :param num_sites: Number of sites from left or right that have been
            traced over.
        :param end: +1 or -1 for tracing over the left or right end
        """
        try:
            return rem_cache[num_sites]
        except KeyError:
            rem = get_remainder(rem_cache, num_sites - 1, end)
            last_pos = num_sites - 1 if end == 1 else -num_sites
            add = np.trace(mpa.lt[last_pos], axis1=1, axis2=2)
            if end == -1:
                rem, add = add, rem

            rem_cache[num_sites] = matdot(rem, add)
            return rem_cache[num_sites]

    num_sites = len(mpa)
    for start, stop in zip(startsites, stopsites):
        # FIXME we could avoid taking copies here, but then in-place
        # multiplication would have side effects. We could make the
        # affected arrays read-only to turn unnoticed side effects into
        # errors.
        ltens = [lten for lten in mpa.lt[start:stop]]
        rem = get_remainder(rem_left, start, 1)
        ltens[0] = matdot(rem, ltens[0])
        rem = get_remainder(rem_right, num_sites - stop, -1)
        ltens[-1] = matdot(ltens[-1], rem)
        yield mp.MPArray(ltens)


def reductions_pmps(pmps, width=None, startsites=None, stopsites=None):
    """Iterate over PMPS partial traces of a PMPS

    `width`, `startsites` and `stopsites`: See
    :func:`reductions_mpo()`.

    :param pmps: Mixed state in locally purified MPS representation
        (PMPS, see :ref:`mpsmpo-definitions`)
    :returns: Iterator over reduced states as PMPS

    """
    startsites, stopsites = \
        _check_reductions_args(len(pmps), width, startsites, stopsites)

    for start, stop in zip(startsites, stopsites):
        pmps.canonicalize(left=start, right=stop)

        # leftmost site
        lten = pmps.lt[start]
        left_bd, system, ancilla, right_bd = lten.shape
        newshape = (1, system, left_bd * ancilla, right_bd)
        ltens = [lten.swapaxes(0, 1).reshape(newshape)]

        # central sites and last site
        ltens += (lten for lten in pmps.lt[start + 1:stop])

        # fix up the last site -- may be the same as the first site
        left_bd, system, ancilla, right_bd = ltens[-1].shape
        newshape = (left_bd, system, ancilla * right_bd, 1)
        ltens[-1] = ltens[-1].reshape(newshape)

        reduced_mps = mp.MPArray(ltens)
        yield reduced_mps


def reductions_mps_as_pmps(mps, width=None, startsites=None, stopsites=None):
    """Iterate over PMPS reduced states of an MPS

    `width`, `startsites` and `stopsites`: See
    :func:`reductions_mpo()`.

    :param mps: Pure state as MPS
    :returns: Iterator over reduced states as PMPS

    """
    pmps = mps_to_pmps(mps)
    return reductions_pmps(pmps, width, startsites, stopsites)


def reductions_mps_as_mpo(mps, width=None, startsites=None, stopsites=None):
    """Iterate over MPO mpdoreduced states of an MPS

    `width`, `startsites` and `stopsites`: See
    :func:`reductions_mpo()`.

    :param mps: Pure state as MPS
    :returns: Iterator over reduced states as MPO

    """
    return map(pmps_to_mpo, reductions_mps_as_pmps(mps, width, startsites,
                                                   stopsites))


def reductions(state, mode, **kwargs):
    """.. todo:: Add docstring"""
    if mode == 'mps':
        return reductions_mps_as_pmps(state, **kwargs), 'pmps'
    elif mode == 'pmps':
        return reductions_pmps(state, **kwargs), 'pmps'
    elif mode == 'mpdo':
        return reductions_mpo(state, **kwargs), 'mpdo'
    else:
        raise ValueError('Unknown mode {!r}'.format(mode))


def pmps_to_mpo(pmps):
    """Convert a local purification MPS to a mixed state MPO.

    A mixed state on n sites is represented in local purification MPS
    form by a MPA with n sites and two physical legs per site. The
    first physical leg is a 'system' site, while the second physical
    leg is an 'ancilla' site.

    :param MPArray pmps: An MPA with two physical legs (system and ancilla)
    :returns: An MPO (density matrix as MPA with two physical legs)

    """
    return mp.dot(pmps, pmps.adj())


def mpo_to_pmps(mpo):
    """Convert a tensor product MPO into a local purification MPS mixed state.

    :param MPArray mpo: An MPA with two physical legs and rank one on all sites
    :returns: An MPA with two physical legs (system and ancilla)

    """
    if not (np.array(mpo.ranks) == 1).all():
        raise ValueError('Only implemented for rank-1 MPOs')
    eigs = (eigh(lten[0, ..., 0] / np.trace(lten[0, ..., 0])) for lten in mpo.lt)
    ltens = (np.sqrt(vals)[None, ...] * vecs for vals, vecs in eigs)
    pmps = mp.MPArray.from_kron(ltens)
    return pmps / mp.norm(pmps)


def mps_to_pmps(mps):
    """Convert a pure MPS into a local purification MPS mixed state.

    The ancilla legs will have dimension one, not increasing the
    memory required for the MPS.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPA with two physical legs (system and ancilla)

    """
    if not (np.array(mps.ndims) == 1).all():
        raise ValueError('Only implemented for unit auxiliary dimensions')
    ltens = (lten.reshape(lten.shape[0:2] + (1, lten.shape[2])) for lten in mps.lt)
    return mp.MPArray(ltens)


def pmps_to_mps(pmps):
    """Convert a PMPS with unit ancilla dimensions to a simple MPS

    If all ancilla dimensions of the PMPS are equal to unity, they are
    removed. Otherwise, an AssertionError is raised.

    """
    assert all(l == 2 for l in pmps.ndims)
    assert all(d[1] == 1 for d in pmps.shape)
    return pmps.reshape([(d[0],) for d in pmps.shape])


def mps_to_mpo(mps):
    """Convert a pure MPS to a mixed state MPO.

    :param MPArray mps: An MPA with one physical leg
    :returns: An MPO (density matrix as MPA with two physical legs)
    """
    return pmps_to_mpo(mps_to_pmps(mps))
