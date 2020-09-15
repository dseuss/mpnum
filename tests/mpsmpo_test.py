# encoding: utf-8
#
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest as pt
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal

import mpnum.factory as factory
import mpnum.mparray as mp
import mpnum.mpsmpo as mm
from mpnum import utils


def _get_reductions(red_fun, mpa, max_red_width):
    startstop = tuple(
        (start, start + width)
        for width in range(1, max_red_width)
        for start in range(len(mpa) - max_red_width + 1)
    )
    start = tuple(start for start, _ in startstop)
    stop = tuple(stop for _, stop in startstop)
    return start, stop, red_fun(mpa, startsites=start, stopsites=stop)


@pt.mark.parametrize('nr_sites, local_dim, rank', [(6, 2, 4), (4, 3, 5)])
def test_pmps_dm_to_array(nr_sites, local_dim, rank, rgen):

    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), rank,
                              randstate=rgen, dtype=np.complex_)
    mpo = mm.pmps_to_mpo(pmps)
    op = mpo.to_array()
    op2 = mm.pmps_dm_to_array(pmps)
    assert_array_almost_equal(op2, op)
    op = mpo.to_array_global()
    op2 = mm.pmps_dm_to_array(pmps, True)
    assert_array_almost_equal(op2, op)


@pt.mark.benchmark(group='pmps_dm_to_array', min_rounds=2)
@pt.mark.parametrize('nr_sites, local_dim, rank', [(10, 2, 4), (10, 2, 16)])
def test_pmps_dm_to_array_fast(nr_sites, local_dim, rank, rgen, benchmark):

    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), rank,
                              dtype=np.complex_, normalized=True,
                              randstate=rgen)
    benchmark(mm.pmps_dm_to_array, pmps)


@pt.mark.benchmark(group='pmps_dm_to_array', min_rounds=2)
@pt.mark.parametrize('nr_sites, local_dim, rank', [(10, 2, 4), (10, 2, 16)])
def test_pmps_dm_to_array_slow(nr_sites, local_dim, rank, rgen, benchmark):

    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), rank,
                              dtype=np.complex_, normalized=True,
                              randstate=rgen)
    benchmark(lambda x: mm.pmps_to_mpo(pmps).to_array(), pmps)


@pt.mark.parametrize(
    'nr_sites, local_dim, rank, keep',
    [(6, 2, 3, (1, 2, 4)), (4, 2, 3, (0, 2)), (4, 2, 3, (1, 3)),
     (4, 2, 3, (1, 2)), (4, 2, 3, (0, 3))])
def test_pmps_reduction(nr_sites, local_dim, rank, keep, rgen):

    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), rank,
                              dtype=np.complex_, normalized=True,
                              randstate=rgen)
    rho = mm.pmps_to_mpo(pmps).to_array_global()
    traceout = [pos for pos in range(nr_sites) if pos not in keep]
    red = utils.partial_trace(rho, traceout)
    pmps_red = mm.pmps_reduction(pmps, keep)
    red2 = mm.pmps_to_mpo(pmps_red).to_array_global()
    red2 = red2.reshape([local_dim] * (2 * len(keep)))
    assert_array_almost_equal(red2, red)


@pt.mark.benchmark(group='pmps_reduction_array')
@pt.mark.parametrize(
    'nr_sites, local_dim, rank, keep',
    [(20, 2, 16, [4, 5, 17, 18]), (20, 2, 32, [4, 5, 17, 18]),
     (32, 2, 16, [1, 2, 17, 18]), (32, 2, 32, [1, 2, 17, 18])]
)
def test_pmps_reduction_array_fast(nr_sites, local_dim, rank, keep, rgen,
                                   benchmark):
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), rank,
                              dtype=np.complex_, normalized=True,
                              randstate=rgen)
    benchmark(lambda: mm.pmps_dm_to_array(mm.pmps_reduction(pmps, keep)))


@pt.mark.benchmark(group='pmps_reduction_array', min_rounds=2)
@pt.mark.parametrize(
    'nr_sites, local_dim, rank, keep',
    [(20, 2, 16, [4, 5, 17, 18]), (20, 2, 32, [4, 5, 17, 18])]
)
def test_pmps_reduction_array_slow_noprune(
        nr_sites, local_dim, rank, keep, rgen, benchmark):
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), rank,
                              dtype=np.complex_, normalized=True,
                              randstate=rgen)
    # NB: The maximal distance between sites of the reduction is
    # limited by the fact that normal numpy builds support arrays with
    # at most 32 indices.
    benchmark(lambda: mm.pmps_to_mpo(mm.pmps_reduction(pmps, keep)).to_array())


@pt.mark.benchmark(group='pmps_reduction_array', min_rounds=2)
@pt.mark.parametrize(
    'nr_sites, local_dim, rank, keep',
    [(32, 2, 16, [1, 2, 17, 18])]
)
def test_pmps_reduction_array_slow_prune(
        nr_sites, local_dim, rank, keep, rgen, benchmark):
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), rank,
                              dtype=np.complex_, normalized=True,
                              randstate=rgen)
    benchmark(
        lambda: mp.prune(mm.pmps_to_mpo(mm.pmps_reduction(pmps, keep)),
                         singletons=True).to_array()
    )


@pt.mark.parametrize(
    'nr_sites, local_dim, rank, keep',
    [(6, 2, 3, (1, 2, 4)), (4, 2, 3, (0, 2)), (4, 2, 3, (1, 3)),
     (4, 2, 3, (1, 2)), (4, 2, 3, (0, 3))])
def test_pmps_reduction_dm_to_array(nr_sites, local_dim, rank, keep, rgen):

    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), rank,
                              dtype=np.complex_, randstate=rgen)
    rho = mm.pmps_to_mpo(pmps).to_array_global()
    traceout = [pos for pos in range(nr_sites) if pos not in keep]
    red = utils.partial_trace(rho, traceout)
    pmps_red = mm.pmps_reduction(pmps, keep)
    red2 = mm.pmps_dm_to_array(pmps_red, True)
    assert_array_almost_equal(red2, red)


@pt.mark.parametrize('nr_sites, local_dim, rank, max_red_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_reductions_mpo(nr_sites, local_dim, rank, max_red_width, rgen):
    mpo = factory.random_mpo(nr_sites, local_dim, rank,
                             randstate=rgen)
    op = mpo.to_array_global()

    start, stop, red = _get_reductions(mm.reductions_mpo, mpo, max_red_width)
    for start, stop, reduced_mpo in zip(start, stop, red):
        # Check that startsites/stopsites and width produce the same result:
        reduced_mpo2 = tuple(mm.reductions_mpo(mpo, stop - start))[start]
        assert_array_almost_equal(reduced_mpo.to_array(),
                                  reduced_mpo2.to_array())
        traceout = tuple(range(start)) + tuple(range(stop, nr_sites))
        red_from_op = utils.partial_trace(op, traceout)
        assert_array_almost_equal(
            reduced_mpo.to_array_global(), red_from_op,
            err_msg="not equal at {}:{}".format(start, stop))

    # check default argument for startsite
    assert len(list(mm.reductions_mpo(mpo, max_red_width))) \
        == nr_sites - max_red_width + 1


@pt.mark.parametrize('nr_sites, local_dim, rank, max_red_width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_reductions_pmps(nr_sites, local_dim, rank, max_red_width, rgen):
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), rank,
                              dtype=np.complex_, randstate=rgen)
    op = mm.pmps_to_mpo(pmps).to_array_global()

    start, stop, red = _get_reductions(mm.reductions_pmps, pmps, max_red_width)
    for start, stop, reduced_pmps in zip(start, stop, red):
        # Check that startsites/stopsites and width produce the same result:
        reduced_pmps2 = tuple(mm.reductions_pmps(pmps, stop - start))[start]
        # NB: reduced_pmps and reduced_pmps2 are in general not equal,
        # but red and red2 are.
        red = mm.pmps_to_mpo(reduced_pmps).to_array_global()
        red2 = mm.pmps_to_mpo(reduced_pmps2).to_array_global()
        assert_array_almost_equal(red, red2)
        traceout = tuple(range(start)) + tuple(range(stop, nr_sites))
        red_from_op = utils.partial_trace(op, traceout)
        assert_array_almost_equal(
            red, red_from_op,
            err_msg="not equal at {}:{}".format(start, stop))

    # check default argument for startsite
    assert len(list(mm.reductions_pmps(pmps, max_red_width))) \
        == nr_sites - max_red_width + 1


@pt.mark.parametrize('nr_sites, local_dim, rank, width',
                     [(6, 2, 4, 3), (4, 3, 5, 2)])
def test_reductions_mps(nr_sites, local_dim, rank, width, rgen):
    mps = factory.random_mps(nr_sites, local_dim, rank, randstate=rgen)
    mpo = mp.localouter(mps, mps.conj())

    pmps_reds = mm.reductions_mps_as_mpo(mps, width)
    mpo_reds = mm.reductions_mpo(mpo, width)

    for red1, red2 in zip(pmps_reds, mpo_reds):
        assert_array_almost_equal(red1.to_array(), red2.to_array())


@pt.mark.parametrize('nr_sites, local_dim, rank', pt.MP_TEST_PARAMETERS)
def test_pmps_to_mpo(nr_sites, local_dim, rank, rgen):
    if (nr_sites % 2) != 0:
        return
    nr_sites = nr_sites // 2
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), rank,
                              dtype=np.complex_, randstate=rgen)
    rho_mp = mm.pmps_to_mpo(pmps).to_array_global()

    # Local form is what we will use: One system site, one ancilla site, etc
    purification = pmps.to_array()
    # Convert to a density matrix
    purification = np.outer(purification, purification.conj())
    purification = purification.reshape((local_dim,) * (2 * 2 * nr_sites))
    # Trace out the ancilla sites
    traceout = tuple(range(1, 2 * nr_sites, 2))
    rho_np = utils.partial_trace(purification, traceout)

    # Here, we need global form
    assert_array_almost_equal(rho_mp, rho_np)


@pt.mark.parametrize('nr_sites, local_dim', [(1, 7), (2, 3), (3, 2), (6, 2),
                                             (4, 3), (5, 2)])
def test_mpo_to_pmps(nr_sites, local_dim, rgen):
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), 1,
                              dtype=np.complex_, randstate=rgen)
    assert_almost_equal(mp.normdist(mm.mpo_to_pmps(mm.pmps_to_mpo(pmps)),
                                    pmps), 0)
    mpo = factory.random_mpa(nr_sites, (local_dim, local_dim), 1,
                             dtype=np.complex_, randstate=rgen, normalized=True)
    assert_almost_equal(mp.normdist(mm.pmps_to_mpo(mm.mpo_to_pmps(mpo)),
                                    mpo), 0)


@pt.mark.parametrize('nr_sites, local_dim, rank', pt.MP_TEST_PARAMETERS)
def test_mps_to_mpo(nr_sites, local_dim, rank, rgen):
    mps = factory.random_mps(nr_sites, local_dim, rank, randstate=rgen)
    # Instead of calling the two functions, we call mps_to_mpo(),
    # which does exactly that:
    #   mps_as_puri = mp.mps_as_local_purification_mps(mps)
    #   mpo = mp.pmps_to_mpo(mps_as_puri)
    mpo = mm.mps_to_mpo(mps)
    # This is also a test of mp.mps_as_local_purification_mps() in the
    # following sense: Local purifications are representations of
    # mixed states. Therefore, compare mps and mps_as_puri by
    # converting them to mixed states.
    state = mps.to_array()
    state = np.outer(state, state.conj())
    state.shape = (local_dim,) * (2 * nr_sites)
    state2 = mpo.to_array_global()
    assert_array_almost_equal(state, state2)
