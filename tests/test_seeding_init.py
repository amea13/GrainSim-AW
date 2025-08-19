# tests/test_seeding_init.py
import numpy as np

from grainsim_aw.core.grid import create_grid, update_ghosts, classify_phases
from grainsim_aw.nucleation.seeding import seed_initialize


def _mk_grid(nx=16, ny=16, dx=1e-6, dy=1e-6, nghost=2):
    cfg_dom = {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "nghost": nghost,
        "bc": "neumann0",
    }
    grid = create_grid(cfg_dom)
    update_ghosts(grid, cfg_dom["bc"])
    return grid


def _center_idx(grid):
    g = grid.nghost
    return g + grid.ny // 2, g + grid.nx // 2


def _count_states(grid):
    masks = classify_phases(grid.fs, grid.nghost)
    return masks, int(masks["mask_sol"].sum()), int(masks["mask_int"].sum())


def test_seed_initialize_single_center_theta_45_deg():
    grid = _mk_grid()
    rng = np.random.default_rng(0)
    init_cfg = {
        "mode": "single_center",
        "theta_deg": 45,
        "random_theta": False,
        "eps_over_edge": 0.02,
        "capture_seed_fs": 0.005,
    }

    placed = seed_initialize(grid, rng, init_cfg)
    assert placed == 1, "应成功放置 1 个核心种子"

    masks, n_sol, n_int = _count_states(grid)
    i0, j0 = _center_idx(grid)

    # 核心必须是固相，且四个对角应为界面
    assert grid.fs[i0, j0] == 1.0
    assert n_sol >= 1
    assert n_int == 4

    expected = {(i0 - 1, j0 - 1), (i0 - 1, j0 + 1), (i0 + 1, j0 - 1), (i0 + 1, j0 + 1)}
    got = set(map(tuple, np.argwhere(masks["mask_int"])))
    assert expected.issubset(got), f"界面应出现在四个对角，但得到 {got}"

    # 被感染元：fs 与 L_dia 的关系、ecc 非零
    for ci, cj in expected:
        assert grid.fs[ci, cj] >= 0.005 - 1e-12
        ecc_norm = float(np.hypot(grid.ecc_x[ci, cj], grid.ecc_y[ci, cj]))
        assert ecc_norm > 0.0

        th = float(grid.theta[ci, cj])
        s, c = abs(np.sin(th)), abs(np.cos(th))
        Lmax = grid.dx / max(s, c, 1e-12)
        assert abs(grid.L_dia[ci, cj] - grid.fs[ci, cj] * Lmax) < 1e-12


def test_seed_initialize_single_center_theta_0_deg():
    grid = _mk_grid()
    rng = np.random.default_rng(0)
    init_cfg = {
        "mode": "single_center",
        "theta_deg": 0,
        "random_theta": False,
        "eps_over_edge": 0.02,
        "capture_seed_fs": 0.005,
    }

    placed = seed_initialize(grid, rng, init_cfg)
    assert placed == 1

    masks, n_sol, n_int = _count_states(grid)
    i0, j0 = _center_idx(grid)

    # 轴向四邻应为界面
    assert grid.fs[i0, j0] == 1.0
    assert n_int == 4

    expected = {(i0, j0 - 1), (i0, j0 + 1), (i0 - 1, j0), (i0 + 1, j0)}
    got = set(map(tuple, np.argwhere(masks["mask_int"])))
    assert expected.issubset(got), f"界面应出现在四个轴向邻胞，但得到 {got}"
