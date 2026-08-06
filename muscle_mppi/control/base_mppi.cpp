#include "base_mppi.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <stdexcept>

static void mujoco_warning_noop(const char*) {}

// Solves the dense linear system A x = b (n x n, row-major A) via Gaussian elimination
// with partial pivoting. A and b are taken by value since both are destructively modified.
static void solve_linear_system(std::vector<double> A, std::vector<double> b,
                                 std::vector<double>& x, int n)
{
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::abs(A[col * n + col]);
        for (int r = col + 1; r < n; ++r) {
            double v = std::abs(A[r * n + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col) {
            for (int c = 0; c < n; ++c) std::swap(A[col * n + c], A[piv * n + c]);
            std::swap(b[col], b[piv]);
        }
        double diag = A[col * n + col];
        for (int r = col + 1; r < n; ++r) {
            double factor = A[r * n + col] / diag;
            if (factor == 0.0) continue;
            for (int c = col; c < n; ++c) A[r * n + c] -= factor * A[col * n + c];
            b[r] -= factor * b[col];
        }
    }
    x.assign(n, 0.0);
    for (int r = n - 1; r >= 0; --r) {
        double sum = b[r];
        for (int c = r + 1; c < n; ++c) sum -= A[r * n + c] * x[c];
        x[r] = sum / A[r * n + r];
    }
}

// Not-a-knot cubic spline through (xk[i], yk[i]), i=0..n-1 with xk strictly increasing.
// Evaluates at integer query points t=0..H-1 into yt. Verified bit-for-bit (to fp precision)
// against scipy.interpolate.CubicSpline(bc_type='not-a-knot') — scipy's default — rather than
// the cheaper natural boundary condition (S''=0 at the ends), so segment shapes match exactly.
static void not_a_knot_cubic_spline(const std::vector<double>& xk, const std::vector<double>& yk,
                                     double* yt, int H)
{
    const int n = static_cast<int>(xk.size());
    if (n == 1) { std::fill(yt, yt + H, yk[0]); return; }
    if (n == 2) {
        const double span = xk[1] - xk[0];
        for (int t = 0; t < H; ++t) {
            const double frac = (span > 0.0) ? (t - xk[0]) / span : 0.0;
            yt[t] = yk[0] + frac * (yk[1] - yk[0]);
        }
        return;
    }

    std::vector<double> h(n - 1);
    for (int i = 0; i < n - 1; ++i) h[i] = xk[i + 1] - xk[i];

    // n == 3 has only one interior knot, so the two not-a-knot conditions (third-derivative
    // continuity at x1 and at x_{n-2} — the same point here) collapse into one equation and
    // the system is rank-deficient. Fall back to natural (S''=0 at both ends) in that case.
    std::vector<double> A(n * n, 0.0), rhs(n, 0.0);
    if (n == 3) {
        A[0] = 1.0; rhs[0] = 0.0;                                  // c0 = 0
        A[1 * n + 0] = h[0]; A[1 * n + 1] = 2.0 * (h[0] + h[1]); A[1 * n + 2] = h[1];
        rhs[1] = 3.0 * ((yk[2] - yk[1]) / h[1] - (yk[1] - yk[0]) / h[0]);
        A[2 * n + 2] = 1.0; rhs[2] = 0.0;                          // c2 = 0
    } else {
        // Not-a-knot at x1: third derivative of segment 0 equals that of segment 1.
        A[0] = -h[1]; A[1] = h[0] + h[1]; A[2] = -h[0]; rhs[0] = 0.0;

        // Interior knots: standard first-derivative-continuity equations.
        for (int i = 1; i <= n - 2; ++i) {
            A[i * n + (i - 1)] = h[i - 1];
            A[i * n + i]       = 2.0 * (h[i - 1] + h[i]);
            A[i * n + (i + 1)] = h[i];
            rhs[i] = 3.0 * ((yk[i + 1] - yk[i]) / h[i] - (yk[i] - yk[i - 1]) / h[i - 1]);
        }

        // Not-a-knot at x_{n-2}: third derivative of the last two segments match.
        A[(n - 1) * n + (n - 3)] = -h[n - 2];
        A[(n - 1) * n + (n - 2)] = h[n - 3] + h[n - 2];
        A[(n - 1) * n + (n - 1)] = -h[n - 3];
        rhs[n - 1] = 0.0;
    }

    std::vector<double> c;
    solve_linear_system(A, rhs, c, n);

    std::vector<double> b(n - 1), d(n - 1);
    for (int j = 0; j < n - 1; ++j) {
        b[j] = (yk[j + 1] - yk[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
    }

    int seg = 0;
    for (int t = 0; t < H; ++t) {
        const double x = static_cast<double>(t);
        while (seg < n - 2 && x > xk[seg + 1]) ++seg;
        const double dx = x - xk[seg];
        yt[t] = yk[seg] + b[seg] * dx + c[seg] * dx * dx + d[seg] * dx * dx * dx;
    }
}

BaseMPPI::BaseMPPI(const TaskConfig& task)
    : task_(task), rng_(std::random_device{}())
{
    mju_user_warning = mujoco_warning_noop;

    if (task_.num_threads > 0)
        omp_set_num_threads(task_.num_threads);

    char error[1000];
    model_ = mj_loadXML(task_.model_path.c_str(), nullptr, error, sizeof(error));
    if (!model_) throw std::runtime_error("Failed to load model: " + std::string(error));

    model_->opt.timestep = task_.dt;

    data_.resize(task_.n_samples + 1);
    for (int i = 0; i <= task_.n_samples; ++i)
        data_[i] = mj_makeData(model_);

    // Detect freejoint (freejoint adds 1 extra qpos DOF via quaternion, so nq != nv).
    has_freejoint_ = (model_->nq != model_->nv);

    // Build actuator → DOF mapping for the controlled joints starting at JOINT_OFFSET.
    for (int j = 0; j < NUM_JOINTS; ++j) {
        int jid = model_->actuator_trnid[2 * (JOINT_OFFSET + j)];
        act_qpos_adr_[j] = model_->jnt_qposadr[jid];
        act_qvel_adr_[j] = model_->jnt_dofadr[jid];
    }

    // Set joint damping to kd_sim so MuJoCo applies it automatically,
    // matching hardware where tau_eff = tau_hill - kd*dq.
    for (int j = 0; j < NUM_JOINTS; ++j)
        model_->dof_damping[act_qvel_adr_[j]] = task_.muscle.kd_sim[j];

    trajectory_.assign(task_.horizon * NUM_MUSCLES, 0.0);
    noise_.assign(task_.n_samples * task_.horizon * NUM_MUSCLES, 0.0);
    costs_.resize(task_.n_samples);

    if (task_.sample_type == "cubic") {
        // Evenly spaced knot timesteps spanning [0, horizon-1], rounded to the nearest
        // integer step (mirrors RTWholeBodyMPPI's perturb_action cubic branch).
        task_.n_knots = std::clamp(task_.n_knots, 2, task_.horizon);
        knot_x_.resize(task_.n_knots);
        for (int k = 0; k < task_.n_knots; ++k) {
            const double idx = static_cast<double>(k) * (task_.horizon - 1)
                              / static_cast<double>(task_.n_knots - 1);
            knot_x_[k] = std::round(idx);
        }
    }
}

BaseMPPI::~BaseMPPI() {
    for (auto* d : data_) mj_deleteData(d);
    mj_deleteModel(model_);
}

void BaseMPPI::sample_noise() {
    if (task_.sample_type == "cubic") {
        sample_noise_cubic();
        return;
    }

    const int H = task_.horizon;

    for (int s = 0; s < task_.n_samples; ++s)
        for (int ti = 0; ti < H; ++ti) {
            for (int j = 0; j < NUM_JOINTS; ++j) {
                const double sigma = task_.noise_sigma_act[j];
                const int base = s * H * NUM_MUSCLES + ti * NUM_MUSCLES + 2 * j;
                noise_[base]     = sigma * normal_(rng_);
                noise_[base + 1] = sigma * normal_(rng_);
            }
        }
}

void BaseMPPI::sample_noise_cubic() {
    const int H = task_.horizon;
    const int K = task_.n_knots;

    std::vector<double> knot_y(K);
    std::vector<double> yt(H);

    for (int s = 0; s < task_.n_samples; ++s) {
        for (int j = 0; j < NUM_JOINTS; ++j) {
            const double sigma = task_.noise_sigma_act[j];
            for (int side = 0; side < 2; ++side) {
                const int m = 2 * j + side;
                for (int k = 0; k < K; ++k) knot_y[k] = sigma * normal_(rng_);
                not_a_knot_cubic_spline(knot_x_, knot_y, yt.data(), H);
                for (int t = 0; t < H; ++t)
                    noise_[s * H * NUM_MUSCLES + t * NUM_MUSCLES + m] = yt[t];
            }
        }
    }
}

void BaseMPPI::warm_start(int n_skip)
{
    const int H = task_.horizon;
    std::vector<double> shifted(H * NUM_MUSCLES);
    for (int t = 0; t < H - n_skip; ++t)
        for (int m = 0; m < NUM_MUSCLES; ++m)
            shifted[t * NUM_MUSCLES + m] = trajectory_[(t + n_skip) * NUM_MUSCLES + m];
    for (int t = H - n_skip; t < H; ++t)
        for (int m = 0; m < NUM_MUSCLES; ++m)
            shifted[t * NUM_MUSCLES + m] = trajectory_[(H - 1) * NUM_MUSCLES + m];
    trajectory_ = std::move(shifted);
}

void BaseMPPI::run_mppi_step(const RobotState& state)
{
    sample_noise();

    #pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < task_.n_samples; ++s)
        costs_[s] = rollout(s, state);

    // Softmin weights over min-max normalised costs.
    // Exclude fall-penalty outliers (≥1e5) from cmax so one falling sample
    // doesn't collapse all non-falling weights to near-uniform.
    static constexpr double kFallThreshold = 1e5;
    double cmin = *std::min_element(costs_.begin(), costs_.end());
    double cmax = cmin;
    for (int s = 0; s < task_.n_samples; ++s)
        if (costs_[s] < kFallThreshold)
            cmax = std::max(cmax, costs_[s]);
    double crange = cmax - cmin;

    std::vector<double> weights(task_.n_samples);
    double wsum = 0.0;
    for (int s = 0; s < task_.n_samples; ++s) {
        double s_hat = (crange > 1e-12)
            ? std::min((costs_[s] - cmin) / crange, 1.0) : 0.0;
        weights[s]   = std::exp(-s_hat / task_.lambda);
        wsum        += weights[s];
    }

    std::vector<double> new_traj(task_.horizon * NUM_MUSCLES, 0.0);
    for (int s = 0; s < task_.n_samples; ++s) {
        double w = weights[s] / wsum;
        for (int t = 0; t < task_.horizon; ++t)
            for (int m = 0; m < NUM_MUSCLES; ++m) {
                int idx = t * NUM_MUSCLES + m;
                new_traj[idx] += w * (trajectory_[idx]
                    + noise_[s * task_.horizon * NUM_MUSCLES + idx]);
            }
    }
    for (int t = 0; t < task_.horizon; ++t)
        for (int m = 0; m < NUM_MUSCLES; ++m) {
            int idx = t * NUM_MUSCLES + m;
            new_traj[idx] = std::clamp(new_traj[idx], action_lo_[m], action_hi_[m]);
        }

    trajectory_ = std::move(new_traj);
}

void BaseMPPI::set_mj_state(mjData* d, const RobotState& state) {
    mj_resetData(model_, d);

    if (has_freejoint_) {
        d->qpos[0] = state.pos[0];
        d->qpos[1] = state.pos[1];
        d->qpos[2] = state.pos[2];
        d->qpos[3] = state.quat[0];
        d->qpos[4] = state.quat[1];
        d->qpos[5] = state.quat[2];
        d->qpos[6] = state.quat[3];

        d->qvel[0] = state.vel[0];
        d->qvel[1] = state.vel[1];
        d->qvel[2] = state.vel[2];
        d->qvel[3] = state.gyro[0];
        d->qvel[4] = state.gyro[1];
        d->qvel[5] = state.gyro[2];
    }

    for (int j = 0; j < NUM_JOINTS; ++j)
        d->qpos[act_qpos_adr_[j]] = state.q[j];

    for (int j = 0; j < NUM_JOINTS; ++j)
        d->qvel[act_qvel_adr_[j]] = state.dq[j];

    mj_forward(model_, d);
}
