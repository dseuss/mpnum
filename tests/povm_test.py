# encoding: utf-8


from __future__ import division, print_function

import itertools as it
from inspect import isfunction

import numpy as np
import pytest as pt
from numpy.testing import (
    assert_almost_equal, assert_array_almost_equal, assert_array_equal)
from six.moves import range, zip, zip_longest
from _pytest.mark import matchmark

import mpnum as mp
import mpnum.povm as povm
import mpnum.povm.mppovm as mppovm
import mpnum.factory as factory
import mpnum.mpsmpo as mpsmpo
from mpnum import _tools

ALL_POVMS = {name: constructor for name, constructor in povm.__dict__.items()
             if name.endswith('_povm') and isfunction(constructor)}

# nr_sites, startsite, local_dim
MPPOVM_PARAM = [
    (4, 0, 2), (5, 0, 3), (5, 1, 2), (6, 1, 2),
    pt.mark.long((7, 1, 2)), pt.mark.long((8, 2, 2)), pt.mark.long((6, 1, 3))
]

# --------------------------------------------------------------
# Many tests in this module have a NON-ZERO FAILURE PROBABILITY.
# --------------------------------------------------------------
#
# It is not guaranteed that probabilities will be estimated to within
# the thresholds used in the tests. Exceptions should have a low
# probability.
#
# TODO: Many tests in this module basically check that "estimation
# error becomes smaller as the number of samples increases". Usually,
# we check that the estimation error is smaller than a constant times
# 1/sqrt(number of samples) (cf. e.g. central limit theorem or
# variance of the estimator). This is a basic consistency check but
# far from a statistical correctness check, which would be desirable.
#
# method, n_samples
MPPOVM_SAMPLE_PARAM = [
    ('direct', 100), ('cond', 100), pt.mark.long(('cond', 1000)),
    ('direct', 2500), pt.mark.long(('direct', 10000)),
    pt.mark.long(('direct', 40000)), pt.mark.long(('direct', 80000))
]


def mp_from_array_repeat(array, nr_sites):
    """Generate a MPA representation of the `nr_sites`-fold tensor product of
    array.
    """
    mpa = mp.MPArray.from_array(array)
    return mp.outer(it.repeat(mpa, nr_sites))


@pt.fixture(params=['random', 'pauli'])
def nopovm(request, local_dim, rgen):
    """Provide different POVMs and non-POVMs for testing

    We provide instances of :class:`povm.localpovm.POVM` with the
    following elements:

    * `pauli`: Generated by :func:`povm.pauli_povm()`

    * `random`: Random (non-Hermitian, non-positive) elements for
      testing. (These elements do not constitute a POVM. We use them
      to distinguish elem.conj() from elem.T in our code.)

    """
    nopovm_name = request.param
    if nopovm_name == 'pauli':
        return povm.pauli_povm(local_dim)
    elif nopovm_name == 'random':
        d = local_dim
        return povm.localpovm.POVM(factory._zrandn((2 * d**2, d, d), rgen))
    else:
        raise ValueError('Unknown fixture name {}'.format(nopovm_name))


@pt.mark.parametrize('dim', [(2), (3), (6), (7)])
def test_povm_normalization_ic(dim):
    for name, constructor in ALL_POVMS.items():
        # Check that the POVM is normalized: elements must sum to the identity
        current_povm = constructor(dim)
        element_sum = sum(iter(current_povm))
        assert_array_almost_equal(element_sum, np.eye(dim))

        # Check that the attribute that says whether the POVM is IC is correct.
        linear_inversion_recons = np.dot(current_povm.linear_inversion_map,
                                         current_povm.probability_map)
        if current_povm.informationally_complete:
            assert_array_almost_equal(
                linear_inversion_recons, np.eye(dim**2),
                err_msg='POVM {} is not informationally complete'.format(name))
        else:
            assert np.abs(linear_inversion_recons - np.eye(dim**2)).max() > 0.1, \
                'POVM {} is informationally complete'.format(name)


@pt.mark.parametrize('nr_sites, local_dim, bond_dim',
                     [(6, 2, 7), (3, 3, 3), (3, 6, 3), (3, 7, 4)])
def test_povm_ic_mpa(nr_sites, local_dim, bond_dim, rgen):
    # Check that the tensor product of the PauliGen POVM is IC.
    paulis = povm.pauli_povm(local_dim)
    inv_map = mp_from_array_repeat(paulis.linear_inversion_map, nr_sites)
    probab_map = mp_from_array_repeat(paulis.probability_map, nr_sites)
    reconstruction_map = mp.dot(inv_map, probab_map)

    eye = factory.eye(nr_sites, local_dim**2)
    assert mp.norm(reconstruction_map - eye) < 1e-5

    # Check linear inversion for a particular example MPA.
    # Linear inversion works for arbitrary matrices, not only for states,
    # so we test it for an arbitrary MPA.
    mpa = factory.random_mpa(nr_sites, local_dim**2, bond_dim, randstate=rgen)
    # Normalize, otherwise the absolute error check below will not work.
    mpa /= mp.norm(mpa)
    probabs = mp.dot(probab_map, mpa)
    recons = mp.dot(inv_map, probabs)
    assert mp.norm(recons - mpa) < 1e-6


@pt.mark.parametrize('local_dim', [(2), (3), (6), (7)])
def test_povm_probability_map(local_dim, nopovm, rgen):
    # Use a random matrix rho for testing (instead of a positive matrix).
    rho = factory._zrandn((local_dim, local_dim), rgen)
    # Compare output from `povm.localpovm.POVM.probability_map` with
    # calculating probabilities element by element.
    probab_direct = np.array([np.trace(np.dot(elem, rho)) for elem in nopovm])
    probab_pmap = np.dot(nopovm.probability_map, rho.ravel())
    assert_array_almost_equal(probab_pmap, probab_direct)


@pt.mark.parametrize('nr_sites, width, local_dim, bond_dim',
                     [(6, 3, 2, 5), (4, 2, 3, 4)])
def test_mppovm_expectation(nr_sites, width, local_dim, bond_dim, nopovm, rgen):
    # Verify that :func:`povm.MPPovm.expectations()` produces
    # correct results.
    pmap = nopovm.probability_map
    mpnopovm = povm.MPPovm.from_local_povm(nopovm, width)
    # Use a random MPO rho for testing (instead of a positive MPO).
    rho = factory.random_mpa(nr_sites, (local_dim,) * 2, bond_dim, rgen)
    reductions = mpsmpo.reductions_mpo(rho, width)
    # Compute expectation values with mpnopovm.expectations(), which
    # uses mpnopovm.probability_map.
    expectations = list(mpnopovm.expectations(rho))
    assert len(expectations) == nr_sites - width + 1

    for evals_mp, rho_red in zip_longest(expectations, reductions):
        # Compute expectation values by constructing each tensor
        # product POVM element.
        rho_red_matrix = rho_red.to_array_global().reshape(
            (local_dim**width,) * 2)
        evals = []
        for factors in it.product(nopovm, repeat=width):
            elem = _tools.mkron(*factors)
            evals.append(np.trace(np.dot(elem, rho_red_matrix)))
        evals = np.array(evals).reshape((len(nopovm),) * width)

        # Compute expectation with a different construction. In the
        # end, this is (should be, we verify it here) equivalent to
        # what `mpnopovm.expectations()` does.
        evals_ten = rho_red.ravel().to_array()
        for _ in range(width):
            evals_ten = np.tensordot(evals_ten, pmap, axes=(0, 1))

        assert_array_almost_equal(evals_ten, evals)
        assert_array_almost_equal(evals_mp.to_array(), evals)


@pt.mark.parametrize(
    'nr_sites, local_dim, bond_dim, startsite, width',
    [(4, 2, 3, 0, 4), (7, 2, 3, 1, 3), (6, (7, 3, 2, 5, 2, 3), 3, 2, 3)]
)
def test_mppovm_embed_expectation(
        nr_sites, local_dim, bond_dim, startsite, width, rgen):
    if hasattr(local_dim, '__iter__'):
        local_dim2 = local_dim
    else:
        local_dim2 = [local_dim] * nr_sites
    local_dim2 = list(zip(local_dim2, local_dim2))

    # Create a local POVM `red_povm`, embed it onto a larger chain
    # (`full_povm`), and go back to the reduced POVM.
    red_povm = mp.outer(
        mp.povm.MPPovm.from_local_povm(mp.povm.pauli_povm(d), 1)
        for d, _ in local_dim2[startsite:startsite + width]
    )
    full_povm = red_povm.embed(nr_sites, startsite, local_dim)
    axes = [(1, 2) if i < startsite or i >= startsite + width else None
            for i in range(nr_sites)]
    red_povm2 = mp.partialtrace(full_povm, axes, mp.MPArray)
    red_povm2 = mp.prune(red_povm2, singletons=True)
    red_povm2 /= np.prod([d for i, (d, _) in enumerate(local_dim2)
                          if i < startsite or i >= startsite + width])
    assert_almost_equal(mp.normdist(red_povm, red_povm2), 0.0)

    # Test with an arbitrary random MPO instead of an MPDO
    mpo = mp.factory.random_mpa(nr_sites, local_dim2, bond_dim, rgen,
                                normalized=True)
    mpo_red = next(mp.reductions_mpo(mpo, width, startsites=[startsite]))
    ept = mp.prune(full_povm.pmf(mpo, 'mpdo'), singletons=True).to_array()
    ept_red = red_povm.pmf(mpo_red, 'mpdo').to_array()
    assert_array_almost_equal(ept, ept_red)


@pt.mark.parametrize('nr_sites, width, local_dim, bond_dim',
                     [(6, 3, 2, 5), (4, 2, 3, 4), (4, 4, 3, 3)])
def test_mppovm_expectation_pure(nr_sites, width, local_dim, bond_dim, rgen):
    paulis = povm.pauli_povm(local_dim)
    mppaulis = povm.MPPovm.from_local_povm(paulis, width)
    psi = factory.random_mps(nr_sites, local_dim, bond_dim, randstate=rgen)
    rho = mpsmpo.mps_to_mpo(psi)
    expect_psi = list(mppaulis.expectations(psi))
    expect_rho = list(mppaulis.expectations(rho))

    assert len(expect_psi) == len(expect_rho)
    for e_rho, e_psi in zip(expect_rho, expect_psi):
        assert_array_almost_equal(e_rho.to_array(), e_psi.to_array())


@pt.mark.parametrize('nr_sites, width, local_dim, bond_dim',
                     [(6, 3, 2, 5), (4, 2, 3, 4)])
def test_mppovm_expectation_pmps(nr_sites, width, local_dim, bond_dim, rgen):
    paulis = povm.pauli_povm(local_dim)
    mppaulis = povm.MPPovm.from_local_povm(paulis, width)
    psi = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                             randstate=rgen)
    rho = mpsmpo.pmps_to_mpo(psi)
    expect_psi = list(mppaulis.expectations(psi, mode='pmps'))
    expect_rho = list(mppaulis.expectations(rho))

    assert len(expect_psi) == len(expect_rho)
    for e_rho, e_psi in zip(expect_rho, expect_psi):
        assert_array_almost_equal(e_rho.to_array(), e_psi.to_array())


@pt.mark.parametrize(
    'nr_sites, local_dim, bond_dim, startsite, width',
    [(4, 2, 3, 0, 4), (4, ((5, 2), (2, 3), (3, 2), (2, 2)), 3, 0, 4),
     (7, 2, 3, 1, 3), (6, ((5, 2), (2, 3), (3, 2), (2, 2), (5, 3), (3, 2)), 3, 2, 3)]
)
def test_mppovm_pmf_as_array_pmps(
        nr_sites, local_dim, bond_dim, startsite, width, rgen):
    if hasattr(local_dim, '__len__'):
        pdims = [d for d, _ in local_dim]
        mppaulis = mp.outer(
            povm.MPPovm.from_local_povm(povm.pauli_povm(d), 1)
            for d in pdims[startsite:startsite + width]
        )
    else:
        pdims = local_dim
        local_dim = (local_dim, local_dim)
        mppaulis = povm.MPPovm.from_local_povm(povm.pauli_povm(pdims), width)
    mppaulis = mppaulis.embed(nr_sites, startsite, pdims)
    pmps = factory.random_mpa(nr_sites, local_dim, bond_dim,
                              randstate=rgen)
    pmps /= mp.norm(pmps)
    rho = mpsmpo.pmps_to_mpo(pmps)
    expect_rho = mppaulis.pmf_as_array(rho, 'mpdo')

    for impl in ['default', 'pmps-ltr', 'pmps-symm']:
        expect_pmps = mppaulis.pmf_as_array(pmps, 'pmps', impl=impl)
        assert_array_almost_equal(expect_rho, expect_pmps, err_msg=impl)


@pt.mark.benchmark(group='pmf_as_array_pmps')
@pt.mark.parametrize(
    'nr_sites, local_dim, bond_dim, startsite, width', [(10, 2, 16, 0, 10)])
@pt.mark.parametrize('impl', ['default', 'pmps-ltr', 'pmps-symm'])
def test_mppovm_pmf_as_array_pmps_benchmark(
        nr_sites, local_dim, bond_dim, startsite, width, impl, rgen, benchmark):
    pauli_y = povm.pauli_parts(local_dim)[1]
    mpp_y = povm.MPPovm.from_local_povm(pauli_y, width) \
                       .embed(nr_sites, startsite, local_dim)
    pmps = factory.random_mpa(nr_sites, (local_dim, local_dim), bond_dim,
                              randstate=rgen)
    pmps /= mp.norm(pmps)
    benchmark(lambda: mpp_y.pmf_as_array(pmps, 'pmps', impl=impl))


@pt.mark.benchmark(group='pmf_as_array_pmps2', min_rounds=2)
@pt.mark.parametrize(
    'nr_sites, local_dim, bond_dim, startsite, width',
    [(32, 2, 20, 10, 6)])
@pt.mark.parametrize('impl', ['default', 'pmps-ltr', 'pmps-symm'])
def test_mppovm_pmf_as_array_pmps_benchmark2(
        nr_sites, local_dim, bond_dim, startsite, width, impl, rgen, benchmark):
    return test_mppovm_pmf_as_array_pmps_benchmark(
        nr_sites, local_dim, bond_dim, startsite, width, impl, rgen, benchmark)


@pt.mark.parametrize(
    'nr_sites, n_small, small_startsite, local_dim',
    [(5, 2, 1, 2), (5, 2, 0, 2), (5, 2, 3, 2),
     (8, 3, 2, 2), (8, 3, 2, 3), (40, 3, 10, 2)])
def test_mppovm_match_elems_local(
        nr_sites, n_small, small_startsite, local_dim, eps=1e-10):
    """Check that match_elems() works for single- and
    multi-Pauli MPPOVMs"""
    n_right = nr_sites - n_small - small_startsite
    assert n_right >= 0

    # "Big" POVM: X on all sites, "small" POVM: X on `n_small` neighbours
    x = povm.x_povm(local_dim)
    big = povm.MPPovm.from_local_povm(x, nr_sites)
    small = povm.MPPovm.from_local_povm(x, n_small) \
                       .embed(nr_sites, small_startsite, local_dim)

    match, prefactors = small.match_elems(big)
    assert match.shape == tuple([len(x)] * n_small * 2)
    assert match.shape == prefactors.shape

    # Verify the expected one-to-one correspondence between POVM elements.
    want = np.eye(np.prod(small.outdims), dtype=bool).reshape(match.shape)
    assert (match == want).all()
    assert (abs(prefactors[match] - 1.0) <= eps).all()
    assert np.isnan(prefactors[~match]).all()

    # "Big" POVM: X on all sites, "small" POVM: Paulis on `n_small` neighbours
    paulis = povm.pauli_povm(local_dim)
    small = povm.MPPovm.from_local_povm(paulis, n_small) \
                       .embed(nr_sites, small_startsite, local_dim)
    match, prefactors = small.match_elems(big)
    assert match.shape == tuple([len(paulis)] * n_small + [len(x)] * n_small)
    assert match.shape == prefactors.shape

    # Verify that the X POVM elements were found where we expect them
    x_pos = tuple([slice(0, len(x))] * n_small)
    want = np.zeros_like(match)
    want[x_pos] = np.eye(len(x)**n_small, dtype=bool).reshape([len(x)] * 2 * n_small)
    assert (match == want).all()
    want = (2 if local_dim > 2 else 3)**-n_small
    assert (abs(prefactors[match] - want) / want <= eps).all()
    assert np.isnan(prefactors[~match]).all()

    # "Big" POVM: Y on all sites, "small" POVM: Paulis on `n_small` neighbours
    y = povm.y_povm(local_dim)
    big = povm.MPPovm.from_local_povm(y, nr_sites)
    match, prefactors = small.match_elems(big)
    assert match.shape == tuple([len(paulis)] * n_small + [len(y)] * n_small)
    assert match.shape == prefactors.shape

    # Verify that the Y POVM elements were found where we expect them
    y_pos = tuple([slice(len(x), len(x) + len(y))] * n_small)
    want = np.zeros_like(match)
    want[y_pos] = np.eye(len(y)**n_small, dtype=bool).reshape([len(y)] * 2 * n_small)
    assert (match == want).all()
    want = (2 if local_dim > 2 else 3)**-n_small
    assert (abs(prefactors[match] - want) / want <= eps).all()
    assert np.isnan(prefactors[~match]).all()


def test_mppovm_match_elems_bell(eps=1e-10):
    """Test match_elems() for a non-product MPPovm"""
    # Four Bell states (basis: |00>, |01>, |10>, |11>)
    bell = np.array((
        [(1/3)**0.5, 0, 0, (1/3)**0.5],   # (0, 0):  |00> + |11>  (proj. weight 1/3)
        [0, 1, 1, 0],                     # (0, 1):  |01> + |10>  (proj. weight 1)
        [(2/3)**0.5, 0, 0, -(2/3)**0.5],  # (0, 2):  |00> - |11>  (proj. weight 2/3)
        [0, 1, -1, 0],                    # (1, 0):  |01> - |10>  (proj. weight 1)
        [(1/3)**0.5, 0, 0, -(1/3)**0.5],  # (1, 1):  |00> - |11>  (proj. weight 1/3)
        [(2/3)**0.5, 0, 0, (2/3)**0.5],   # (1, 2):  |00> + |11>  (proj. weight 2/3)
    )) / 2**0.5
    bell_proj = np.einsum('ij, ik -> ijk', bell, bell.conj())
    bell_proj = bell_proj.reshape((2, 3) + (2,) * 4)
    # Four Bell states and two product states
    vecs = np.array((
        [0, 1, -1, 0],            # (0, 0):  |01> - |10>  (proj. weight 0.5)
        [0, 2**0.5, 0, 0],        # (0, 1):  |01>         (proj. weight 0.5)
        [0, 0, 2**0.5, 0],        # (1, 0):  |10>         (proj. weight 0.5)
        [2**0.5, 0, 0, -2**0.5],  # (1, 1):  |00> - |11>  (proj. weight 1)
        [0, 1, 1, 0],             # (2, 0):  |01> + |10>  (proj. weight 0.5)
        [2**0.5, 0, 0, 2**0.5],   # (2, 1):  |00> + |11>  (proj. weight 1)
    )) / 2
    proj = np.einsum('ij, ik -> ijk', vecs, vecs.conj())
    proj = proj.reshape((3,) + (2,) * 5)

    # Big POVM: The four Bell states (repeated two times)
    big = povm.MPPovm.from_array_global(bell_proj, plegs=3)
    big = mp.outer([big, big])
    # Small POVM: Two of the Bell states and four product states (on
    # the last two sites)
    small = povm.MPPovm.from_array_global(proj, plegs=3).embed(4, 2, 2)

    # Check that the POVM is normalized: elements must sum to the identity
    for mppovm in big, small:
        element_sum = sum(x.to_array_global().reshape(16, 16)
                          for x in mppovm.elements)
        assert_array_almost_equal(element_sum, np.eye(16))

    match, prefactors = small.match_elems(big, eps=eps)
    # Verify the correspondence which can be read off above
    want = np.zeros((3, 2, 2, 3), dtype=bool)
    want[0, 0, 1, 0] = True  # |01> - |10>
    want[2, 0, 0, 1] = True  # |01> + |10>
    want[1, 1, 0, 2] = True  # |00> - |11>
    want[1, 1, 1, 1] = True  # |00> - |11>
    want[2, 1, 0, 0] = True  # |00> + |11>
    want[2, 1, 1, 2] = True  # |00> + |11>
    assert (match == want).all()
    assert abs(prefactors[0, 0, 1, 0] - 0.5) <= eps
    assert abs(prefactors[2, 0, 0, 1] - 0.5) <= eps
    assert abs(prefactors[1, 1, 0, 2] - 1.5) <= eps
    assert abs(prefactors[1, 1, 1, 1] - 3) <= eps
    assert abs(prefactors[2, 1, 0, 0] - 3) <= eps
    assert abs(prefactors[2, 1, 1, 2] - 1.5) <= eps
    assert np.isnan(prefactors[~match]).all()


@pt.mark.parametrize('method, n_samples',
                     MPPOVM_SAMPLE_PARAM + [pt.mark.verylong(('cond', 10000))])
@pt.mark.parametrize('nr_sites, startsite, local_dim', MPPOVM_PARAM)
def test_mppovm_sample(
        method, n_samples, nr_sites, startsite, local_dim, rgen):
    """Check that probability estimates from samples are reasonable accurate"""
    bond_dim = 3
    eps = 1e-10
    mps = factory.random_mps(nr_sites, local_dim, bond_dim, rgen)
    mps.normalize()

    local_x = povm.x_povm(local_dim)
    local_y = povm.y_povm(local_dim)
    xx = povm.MPPovm.from_local_povm(local_x, 2)
    y = povm.MPPovm.from_local_povm(local_y, 1)
    mpp = mp.outer([xx, povm.MPPovm.eye([local_dim]), y]) \
            .embed(nr_sites, startsite, local_dim)

    pmf_exact = mpp.pmf_as_array(mps, 'mps', eps)

    if n_samples > 100:
        n_gr = 5
    elif local_dim == 3:
        n_gr = 2
    else:
        n_gr = 3
    samples = mpp.sample(rgen, mps, n_samples, method, n_gr, 'mps', eps=eps)

    pmf_est = mpp.est_pmf(samples)

    assert abs(pmf_est.sum() - 1.0) <= eps
    assert abs(pmf_exact - pmf_est).max() <= 3 / n_samples**0.5


@pt.mark.parametrize('method, n_samples', MPPOVM_SAMPLE_PARAM)
@pt.mark.parametrize('nr_sites, startsite, local_dim', MPPOVM_PARAM)
def test_mppovm_est_pmf_from(
        method, n_samples, nr_sites, startsite, local_dim, rgen):
    """Check that probability estimates from samples are reasonable accurate"""
    bond_dim = 3
    eps = 1e-10
    nr_small = 4
    mps = factory.random_mps(nr_sites, local_dim, bond_dim, rgen)
    mps.normalize()

    lx = povm.x_povm(local_dim)
    ly = povm.y_povm(local_dim)
    lp = povm.pauli_povm(local_dim)
    x = povm.MPPovm.from_local_povm(lx, 1)
    y = povm.MPPovm.from_local_povm(ly, 1)
    pauli = povm.MPPovm.from_local_povm(lp, 1)
    xy = mp.outer((x, y))
    mpp = mp.outer((xy,) * (nr_sites // 2))
    if (nr_sites % 2) == 1:
        mpp = mp.outer((mpp, x))
    small_mpp = mp.outer((pauli, povm.MPPovm.eye([local_dim]), pauli, pauli)) \
                  .embed(nr_sites, startsite, local_dim)

    x_given = np.arange(len(lp)) < len(lx)
    y_given = (np.arange(len(lp)) >= len(lx)) \
              & (np.arange(len(lp)) < len(lx) + len(ly))
    given_sites = [x_given if ((startsite + i) % 2) == 0 else y_given
                   for i in (0, 2, 3)]
    given_expected = np.einsum('i, j, k -> ijk', *given_sites)
    pmf_exact = small_mpp.pmf_as_array(mps, 'mps', eps)

    if n_samples > 100:
        n_gr = 5
    elif local_dim == 3:
        n_gr = 2
    else:
        n_gr = 3

    samples = mpp.sample(rgen, mps, n_samples, method, n_gr, 'mps', eps=eps)
    est_pmf, est_n_samples = small_mpp.est_pmf_from(mpp, samples)
    # In this case, we use all the samples from `mpp`.
    assert est_n_samples == n_samples
    given = ~np.isnan(est_pmf)
    assert (given == given_expected).all()

    assert abs(pmf_exact[given].sum() - est_pmf[given].sum()) <= eps
    assert abs(pmf_exact[given] - est_pmf[given]).max() <= 1 / n_samples**0.5


@pt.mark.parametrize('method, n_samples', MPPOVM_SAMPLE_PARAM)
@pt.mark.parametrize('nr_sites, startsite, local_dim', MPPOVM_PARAM)
def test_mppovm_est(
        method, n_samples, nr_sites, startsite, local_dim, rgen):
    """Check that estimates from .est_pmf() and .est_lfun() are reasonably
    accurate

    """
    bond_dim = 3
    eps = 1e-10
    mps = factory.random_mps(nr_sites, local_dim, bond_dim, rgen)
    mps.normalize()

    local_x = povm.x_povm(local_dim)
    local_y = povm.y_povm(local_dim)
    xx = povm.MPPovm.from_local_povm(local_x, 2)
    y = povm.MPPovm.from_local_povm(local_y, 1)
    mpp = mp.outer([xx, povm.MPPovm.eye([local_dim]), y]) \
            .embed(nr_sites, startsite, local_dim)

    p_exact = mpp.pmf_as_array(mps, 'mps', eps)
    p_exact = _tools.check_pmf(p_exact, eps, eps)

    cov_p_exact = np.diag(p_exact.flat) - np.outer(p_exact.flat, p_exact.flat)
    samples = mpp.sample(rgen, mps, n_samples, method, 4, 'mps', eps=eps)

    p_est = mpp.est_pmf(samples)
    ept, cov = mpp.est_lfun(None, None, samples, None, eps)
    ept_ex, single_cov_ex = mpp.lfun(None, None, mps, 'mps', eps)
    # The two exact values must match
    assert abs(ept_ex - p_exact.ravel()).max() <= eps
    # The two exact values must match
    assert abs(cov_p_exact - single_cov_ex).max() <= eps
    # The two estimates must match. This verifies that we have chosen
    # our estimator will be unbiased. (There are many other things we
    # might want to know about our estimator.)
    assert (ept == p_est.ravel()).all()
    # The estimate must be close to the true value
    assert abs(p_exact - p_est).max() <= 3 / n_samples**0.5

    cov_ex = cov_p_exact / n_samples
    # The covariances of the sample means (which we estimate here)
    # decrease by 1/n_samples, so we multiply with n_samples before
    # comparing to the rule-of-thumb for the estimation error.
    assert abs(cov - cov_ex).max() * n_samples <= 1 / n_samples**0.5

    funs = []
    nsoutdims = mpp.nsoutdims
    out = np.unravel_index(range(np.prod(nsoutdims)), nsoutdims)
    out = np.array(out).T[:, None, :].copy()
    for ind in range(np.prod(nsoutdims)):
        funs.append(lambda s, ind=ind: (s == out[ind]).all(1))

    # All probabilities sum to one, and we can estimate that well.
    coeff = np.ones(len(funs), dtype=float)
    # Test with dummy weights
    weights = np.ones(n_samples, dtype=float)
    sum_ept, sum_var = mpp.est_lfun(coeff, funs, samples, weights, eps)
    assert abs(sum_ept - 1.0) <= eps
    assert sum_var <= eps

    # Check a sum of probabilities with varying signs.
    coeff = ((-1)**rgen.choice(2, len(funs))).astype(float)
    sum_ept, sum_var = mpp.est_lfun(coeff, funs, samples, None, eps)
    ex_sum = np.inner(coeff, p_exact.flat)
    ex_var = np.inner(coeff, np.dot(cov_ex, coeff))
    assert abs(sum_ept - ex_sum) <= 3 / n_samples**0.5
    assert abs(sum_var - ex_var) * n_samples <= 3 / n_samples**0.5

    # Convert samples to counts and test again
    counts = mpp.est_pmf(samples, normalize=False, eps=eps)
    assert counts.sum() == n_samples
    count_samples = np.array(np.unravel_index(range(np.prod(mpp.nsoutdims)),
                                              mpp.nsoutdims)).T
    weights = counts.ravel()
    sum_ept2, sum_var2 = mpp.est_lfun(coeff, funs, count_samples, weights, eps)
    assert abs(sum_ept - sum_ept2) <= eps
    assert abs(sum_var - sum_var2) <= eps


@pt.mark.parametrize(
    'method, n_samples', [
        ('direct', 100),
    ])
@pt.mark.parametrize(
    'nr_sites, local_dim, bond_dim, measure_width', [
        (3, 3, 2, 2),
        (3, 2, 3, 3),
        (5, 2, 3, 2),
    ])
def test_mppovmlist_pack_unpack_samples(
        method, n_samples, nr_sites, local_dim, bond_dim, measure_width,
        rgen, eps=1e-10):
    """Check that packing and unpacking samples does not change them"""

    mps = factory.random_mps(nr_sites, local_dim, bond_dim, rgen)
    mps.normalize()

    s_povm = povm.pauli_mpp(measure_width, local_dim).block(nr_sites)
    samples = tuple(s_povm.sample(
        rgen, mps, n_samples, method, mode='mps', pack=False, eps=eps))
    packed = tuple(s_povm.pack_samples(samples))
    unpacked = tuple(s_povm.unpack_samples(packed))

    assert all(s.dtype == np.uint8 for s in samples)
    assert all(s.dtype == np.uint8 for s in unpacked)
    assert all((s == u).all() for s, u in zip(samples, unpacked))


def _pytest_want_long(request):
    # FIXME: Is there a better way to find out whether items marked
    # with `long` should be run or not?
    class dummy:
        keywords = {'long': pt.mark.verylong}
    return matchmark(dummy, request.config.option.markexpr)

@pt.fixture(params=[False, True])
def splitpauli(n_samples, nonuniform, request):
    # We use this fixture to skip certain value combinations for
    # non-long tests.
    #
    # FIXME: Is there a better way to select certain value
    # combinations from the different pt.mark.parametrize() decorators
    # except for writing down all combinations by hand?
    splitpauli = request.param
    if (not splitpauli) or _pytest_want_long(request) \
       or (n_samples >= 10000 and nonuniform):
        return splitpauli
    pt.skip("Should only be run in long tests")
    return None

@pt.mark.parametrize(
    'method, n_samples', [
        ('direct', 1000), ('direct', 10000),
        pt.mark.verylong(('direct', 100000)), pt.mark.verylong(('cond', 100))
    ])
@pt.mark.parametrize(
    'nr_sites, local_dim, bond_dim, measure_width, local_width', [
        (4, 2, 3, 2, 2),
        pt.mark.verylong((5, 2, 2, 3, 2)),
        pt.mark.verylong((4, 3, 2, 2, 2)),
    ])
@pt.mark.parametrize('nonuniform', [False, True])
def test_mppovmlist_est_pmf_from(
        method, n_samples, nr_sites, local_dim, bond_dim, measure_width,
        local_width, nonuniform, splitpauli, rgen, eps=1e-10):
    """Verify that estimated probabilities from MPPovmList.est_pmf_from()
    are reasonable accurate

    """

    mps = factory.random_mps(nr_sites, local_dim, bond_dim, rgen)
    mps.normalize()

    x, y = (povm.MPPovm.from_local_povm(p, 1)
            for p in povm.pauli_parts(local_dim)[:2])
    # POVM list with global support
    g_povm = povm.pauli_mpps(measure_width, local_dim).repeat(nr_sites)
    if nonuniform:
        add_povm = mp.outer((nr_sites - 1) * (x,) + (y,))
        g_povm = povm.MPPovmList(g_povm.mpps + (add_povm,))
    # POVM list with local support
    l_povm = povm.pauli_mpps if splitpauli else povm.pauli_mpp
    l_povm = l_povm(local_width, local_dim).block(nr_sites)
    samples = tuple(g_povm.sample(
        rgen, mps, n_samples, method, mode='mps', eps=eps))
    est_prob, n_samples = zip(*l_povm.est_pmf_from(g_povm, samples, eps))
    exact_prob = tuple(l_povm.pmf_as_array(mps, 'mps', eps))
    # Consistency check on n_samples: All entries should be equal
    # unless `nonuniform` is True.
    all_n_sam = np.concatenate(n_samples)
    assert (not (all_n_sam == all_n_sam[0]).all()) == nonuniform
    for n_sam, est, exact, mpp in zip(
            n_samples, est_prob, exact_prob, l_povm.mpps):
        assert est.shape == mpp.nsoutdims
        assert est.shape == exact.shape
        assert n_sam.shape == exact.shape
        # Compare against exact probabilities
        assert (abs(est - exact) / (3 / n_sam**0.5)).max() <= 1


def _get_povm(name, nr_sites, local_dim, local_width):
    if name == 'global':
        return povm.pauli_mpps(local_width, local_dim).repeat(nr_sites)
    elif name == 'splitpauli':
        return povm.pauli_mpps(local_width, local_dim).block(nr_sites)
    elif name == 'pauli':
        return povm.pauli_mpp(local_width, local_dim).block(nr_sites)
    elif name == "all-y":
        return povm.MPPovmList([povm.MPPovm.from_local_povm(
            povm.pauli_parts(local_dim)[1], nr_sites)])
    elif name == "local-x":
        return povm.MPPovmList([
            povm.MPPovm.from_local_povm(
                povm.pauli_parts(local_dim)[0], local_width)
            .embed(nr_sites, 0, local_dim)
        ])
    else:
        raise ValueError('Unknown MP-POVM list {!r}'.format(name))

POVM_COMBOS = [
    ('global', 'pauli'), ('splitpauli', 'pauli'), ('pauli', 'pauli'),
    ('global', 'all-y'), ('all-y', 'local-x'), ('all-y', 'pauli'),
    pt.mark.verylong(('splitpauli', 'splitpauli')), ('pauli', 'splitpauli')
]
POVM_IDS = ['+'.join(getattr(x, 'args', (x,))[0]) for x in POVM_COMBOS]

@pt.fixture(params=POVM_COMBOS, ids=POVM_IDS)
def povm_combo(function, request):
    # We use this fixture to skip certain value combinations for
    # non-long tests.
    #
    # FIXME: Is there a better way to select certain value
    # combinations from the different pt.mark.parametrize() decorators
    # except for writing down all combinatiosn by hand?
    combo = request.param
    if _pytest_want_long(request):
        return combo
    if function == 'randn' or combo == ('global', 'pauli'):
        return combo
    pt.skip("Should only be run in long tests")
    return None

@pt.mark.parametrize(
    'method, n_samples', [
        pt.mark.verylong(('cond', 100)),
        ('direct', 1000),
        pt.mark.verylong(('direct', 100000)),
    ])
@pt.mark.parametrize(
    'nr_sites, local_dim, bond_dim, measure_width, local_width', [
        (3, 2, 3, 2, 2),
        pt.mark.verylong((4, 2, 3, 3, 2)),
        pt.mark.verylong((5, 2, 3, 2, 2)),
    ])
@pt.mark.parametrize('nonuniform', [True, pt.mark.verylong(False)])
@pt.mark.parametrize('function',
                     ['randn', 'ones', 'signs', pt.mark.verylong('rand')])
def test_mppovmlist_est_lfun_from(
        method, n_samples, nr_sites, local_dim, bond_dim, measure_width,
        local_width, nonuniform, function, povm_combo, rgen, eps=1e-10):
    """Verify that estimated probabilities from MPPovmList.est_pmf_from()
    are reasonable accurate

    .. todo:: This test is too long and should be split into several
              smaller tests. Also, some of the testing done here is
              redundant.

    """

    mps = factory.random_mps(nr_sites, local_dim, bond_dim, rgen)
    mps.normalize()

    sample_povm, fun_povm = povm_combo
    estimation_impossible = sample_povm == "all-y" and \
                            fun_povm in {"local-x", "pauli"}
    fromself = sample_povm == fun_povm and measure_width == local_width
    # s_povm: POVM used to obtain samples
    s_povm = _get_povm(sample_povm, nr_sites, local_dim, measure_width)
    # f_povm: POVM on which a linear function is defined
    if fromself:
        f_povm = s_povm
    else:
        f_povm = _get_povm(fun_povm, nr_sites, local_dim, local_width)
    if function == 'rand':
        coeff = lambda x: rgen.rand(*x)
    elif function == 'randn':
        coeff = lambda x: rgen.randn(*x)
    elif function == 'ones':
        coeff = lambda x: np.ones(x)
    elif function == 'signs':
        coeff = lambda x: rgen.choice([1., -1.], x)
    else:
        raise ValueError('Unknown function {!r}'.format(function))

    # More POVMs in s_povm means more samples. Consider this in the tests.
    n_samples_eff = n_samples * len(s_povm.mpps)
    # We divide the coefficients by len(f_povm.mpps) to make the
    # estimated value have approximately same magnitude, independently
    # of len(f_povm.mpps).
    coeff = [coeff(mpp.nsoutdims) / len(f_povm.mpps) for mpp in f_povm.mpps]
    samples = tuple(s_povm.sample(
        rgen, mps, n_samples, method, mode='mps', eps=eps))
    exact_prob = tuple(f_povm.pmf_as_array(mps, 'mps', eps))

    # Compute exact estimate directly
    exact_est1, exact_var1 = f_povm.lfun([c.ravel() for c in coeff], None, mps, 'mps', eps)
    # Compute exact estimate and variance using the other POVM
    exact_est2, exact_var2 = f_povm.lfun_from(s_povm, coeff, mps, 'mps', eps=eps)
    if estimation_impossible:
        assert np.isnan(exact_est2)
        assert np.isnan(exact_var2)
    else:
        # Estimates must agree.
        assert abs(exact_est1 - exact_est2) <= eps
        if fromself:
            # Variances can be different unless f_povm and s_povm are the same.
            assert abs(exact_var1 - exact_var2) <= eps

    est, var = f_povm.est_lfun_from(s_povm, coeff, samples, eps)

    if fromself:
        # In this case, est_lfun() and est_lfun_from() must give exactly
        # the same result.
        est2, var2 = f_povm.est_lfun([c.ravel() for c in coeff],
                                    None, samples, eps)
        assert abs(est - est2) <= eps
        assert abs(var - var2) <= eps
        # We use est_pmf() to test est_pmf_from()
        # again. MPPovmList.est_pmf() just aggregates results from
        # MPPovm.est_pmf().
        pmf1 = f_povm.est_pmf(samples, normalized=True, eps=eps)
        pmf2, _ = zip(*f_povm.est_pmf_from(s_povm, samples, eps=eps))
        assert all(abs(p1 - p2).max() <= eps for p1, p2 in zip(pmf1, pmf2))

    # The final estimator is based on the samples for
    # `s_povm`. Therefore, it is correct to use `n_samples_eff` below
    # (and not the "effective samples" for the `f_povm` probability
    # estimation returned by :func:`f_povm.est_pmf_from()`).
    exact_est = sum(np.inner(c.flat, p.flat) for c, p in zip(coeff, exact_prob))
    if estimation_impossible:
        assert np.isnan(est)
    else:
        assert abs(exact_est - exact_est2) <= eps
        bound = 20 if fun_povm == 'all-y' else 6
        assert abs(est - exact_est) <= bound / n_samples_eff**0.5
        if function == 'ones':
            assert abs(exact_est - 1) <= eps
            assert abs(est - exact_est) <= eps

    # The code below will only work for small systems. Probably
    # nr_sites = 16 will work, but let's stay safe.
    assert nr_sites <= 8, "Larger systems will require a lot of memory"

    # Use the estimator from `f_povm._estfun_from_estimator()` to
    # compute the exact variance of the estimate. We can assume that
    # estimator is mostly correct because we have checked that it
    # produces accurate estimates (for large numbers of samples)
    # above.
    #
    # FIXME: Drop the exact variance computation here and use
    # exact_var2 from above.
    #
    # Convert from matching functions + coefficients to coefficients
    # for each probability.
    n_samples2 = [s.shape[0] for s in samples]
    _, est_coeff, est_funs = f_povm._lfun_estimator(s_povm, coeff, n_samples2, eps)
    est_p_coeff = [np.zeros(mpp.nsoutdims, float) for mpp in s_povm.mpps]
    for fun_coeff, funs, p_coeff, mpp in zip(
            est_coeff, est_funs, est_p_coeff, s_povm.mpps):
        out = np.unravel_index(range(np.prod(mpp.nsoutdims)), mpp.nsoutdims)
        out = np.array(out).T.copy()
        for c, fun in zip(fun_coeff, funs):
            match = fun(out)
            p_coeff.flat[match] += c
    exact_prob = tuple(s_povm.pmf_as_array(mps, 'mps', eps))
    exact_p_cov = (np.diag(p.flat) - np.outer(p.flat, p.flat) for p in exact_prob)
    exact_var = sum(np.inner(c.flat, np.dot(cov, c.flat))
                    for c, cov in zip(est_p_coeff, exact_p_cov))
    if estimation_impossible:
        assert np.isnan(var)
    else:
        assert abs(exact_var - exact_var2) <= eps
        if fromself:
            # `f_povm` and `s_povm` are equal. We must obtain exactly the
            # same result without using the matching functions from above:
            exact_prob = tuple(f_povm.pmf_as_array(mps, 'mps', eps))
            exact_p_cov = (np.diag(p.flat) - np.outer(p.flat, p.flat)
                           for p in exact_prob)
            exact_var2 = sum(np.inner(c.flat, np.dot(cov, c.flat))
                             for c, cov in zip(coeff, exact_p_cov))
            assert abs(exact_var - exact_var2) <= eps
        # Convert variance to variance of the estimator (=average)
        exact_var /= n_samples

        if sample_povm == 'pauli':
            bound = 6
        elif fun_povm == 'all-y':
            bound = 10
        else:
            bound = 1
        assert n_samples * abs(var - exact_var) <= bound / n_samples_eff**0.5
        if function == 'ones':
            assert abs(exact_var) <= eps
            assert abs(var - exact_var) <= eps
