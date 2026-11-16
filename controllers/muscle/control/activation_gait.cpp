#include "activation_gait.h"

#include <charconv>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <mujoco/mujoco.h>
#include <unistd.h>

#include "muscle.h"
#include "../../common/gait.h"      // gait_table_path()
#include "../../common/harness.h"   // joint_addresses()

// Bump when the generation algorithm changes, so existing files regenerate.
static constexpr int kGeneratorVersion = 1;

// Suspended (fixed-base) Go2: bias torques without contact, as the Python generator uses.
static const char* kSuspendedModel = "unitree_mujoco/unitree_robots/go2/scene_suspended.xml";

// 64-bit FNV-1a: the fingerprint hash of the muscle parameters and the gait table.
static constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
static constexpr uint64_t kFnvPrime  = 1099511628211ULL;

static void fnv1a(uint64_t& h, const std::string& bytes)
{
    for (unsigned char c : bytes) { h ^= c; h *= kFnvPrime; }
}

static std::string hex16(uint64_t h)
{
    char out[17];
    std::snprintf(out, sizeof out, "%016" PRIx64, h);
    return out;
}

std::string activation_gait_path(const std::string& label)
{
    return "controllers/muscle/gaits/activation_gait_" + label + ".tsv";
}

// Hash of the table file's bytes, so regenerating a table under the same label
// also regenerates its activation gait.
static std::string table_fingerprint(const std::string& table_abs)
{
    std::ifstream f(table_abs, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open gait table " + table_abs
                                     + " (write it with gait_gen/generate.py)");
    std::ostringstream bytes;
    bytes << f.rdbuf();

    uint64_t h = kFnvOffset;
    fnv1a(h, bytes.str());
    return hex16(h);
}

std::string muscle_fingerprint(const MuscleParams& p)
{
    // FNV-1a over the parameters as exact round-trip text ("%.17g"), so the hash
    // is the same on any platform that parses the YAML to the same doubles.
    uint64_t h = kFnvOffset;
    auto feed = [&h](const std::string& s) { fnv1a(h, s); };
    auto feed_value = [&feed](double v) {
        char buf[40];
        std::snprintf(buf, sizeof buf, "%.17g;", v);
        feed(buf);
    };
    auto feed_array = [&](const char* name, const double* a) {
        feed(name);
        for (int j = 0; j < NUM_JOINTS; ++j) feed_value(a[j]);
    };
    feed("v" + std::to_string(kGeneratorVersion) + ";");
    feed("stiffness"); feed_value(p.stiffness);
    feed_array("lce_min", p.lce_min);     feed_array("lce_max", p.lce_max);
    feed_array("phi_min", p.phi_min);     feed_array("phi_max", p.phi_max);
    feed_array("pFLmax", p.pFLmax);       feed_array("FVmax", p.FVmax);
    feed_array("vmax", p.vmax);           feed_array("peak_force", p.peak_force);
    // Hill ablation switches change hill_invert_torque, so they change the gait.
    // Fed only when a component is off, so full-model gaits keep their existing
    // fingerprint. activation_dynamics isn't used by gait generation.
    if (!(p.use_fl && p.use_fv && p.use_passive)) {
        feed("ablation");
        feed_value(p.use_fl); feed_value(p.use_fv); feed_value(p.use_passive);
    }

    return hex16(h);
}

std::string activation_gait_header(const std::string& label, const std::string& table,
                                   const MuscleParams& p)
{
    char stiff[40];   // shortest text that reads back as the same double (e.g. "0.8")
    const auto res = std::to_chars(stiff, stiff + sizeof stiff, p.stiffness);
    return "activation_gait source=" + label + " table=" + table
           + " stiffness=" + std::string(stiff, res.ptr) + " muscle=" + muscle_fingerprint(p);
}

// Rows of whitespace-separated numbers; lines starting with '#' and blank lines are skipped.
static std::vector<std::vector<double>> read_tsv(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open gait source: " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<double> row;
        std::istringstream ss(line);
        double v;
        while (ss >> v) row.push_back(v);
        if (!row.empty()) rows.push_back(std::move(row));
    }
    return rows;
}

static std::string first_line(const std::string& path)
{
    std::ifstream f(path);
    std::string line;
    if (f) std::getline(f, line);
    return line;
}

static void generate(const std::string& header, const MuscleParams& p,
                     const std::string& src_abs, const std::string& out_abs)
{
    const std::vector<std::vector<double>> src = read_tsv(src_abs);
    if ((int)src.size() != 2 * NUM_JOINTS)
        throw std::runtime_error("Gait source " + src_abs + ": expected "
                                 + std::to_string(2 * NUM_JOINTS) + " rows, got " + std::to_string(src.size()));
    const size_t n_phases = src[0].size();
    for (const auto& r : src)
        if (r.size() != n_phases) throw std::runtime_error("Gait source " + src_abs + ": inconsistent row lengths");

    char err[1000];
    const std::string model_path = repo_path(kSuspendedModel);
    mjModel* m = mj_loadXML(model_path.c_str(), nullptr, err, sizeof err);
    if (!m) throw std::runtime_error("Failed to load " + model_path + ": " + err);
    mjData* d = mj_makeData(m);
    int qa[NUM_JOINTS], qv[NUM_JOINTS];
    joint_addresses(m, qa, qv);

    // out[row][t]: rows 0..11 = a1 (agonist), rows 12..23 = a2 (antagonist)
    std::vector<std::vector<double>> out(2 * NUM_JOINTS, std::vector<double>(n_phases));
    for (size_t t = 0; t < n_phases; ++t) {
        mj_resetData(m, d);
        for (int j = 0; j < NUM_JOINTS; ++j) {
            d->qpos[qa[j]] = src[j][t];
            d->qvel[qv[j]] = src[NUM_JOINTS + j][t];
        }
        mj_forward(m, d);
        for (int j = 0; j < NUM_JOINTS; ++j)
            hill_invert_torque(src[j][t], src[NUM_JOINTS + j][t], d->qfrc_bias[qv[j]], j,
                               p, p.stiffness, out[j][t], out[NUM_JOINTS + j][t]);
    }
    mj_deleteData(d);
    mj_deleteModel(m);

    // Write beside the target and rename into place (atomic on the same filesystem).
    std::filesystem::create_directories(std::filesystem::path(out_abs).parent_path());
    const std::string tmp = out_abs + ".tmp." + std::to_string(getpid());
    {
        FILE* f = std::fopen(tmp.c_str(), "w");
        if (!f) throw std::runtime_error("Cannot write " + tmp);
        std::fprintf(f, "# %s\n", header.c_str());
        for (const auto& row : out)   // same layout/format as np.savetxt(delimiter='\t', fmt='%.8f')
            for (size_t t = 0; t < n_phases; ++t)
                std::fprintf(f, "%.8f%c", row[t], t + 1 < n_phases ? '\t' : '\n');
        if (std::fclose(f) != 0) throw std::runtime_error("Error writing " + tmp);
    }
    std::filesystem::rename(tmp, out_abs);
}

bool ensure_activation_gait(const std::string& label, const MuscleParams& p)
{
    const std::string table_abs = repo_path(gait_table_path(label));
    const std::string out_abs   = repo_path(activation_gait_path(label));
    const std::string header    = activation_gait_header(label, table_fingerprint(table_abs), p);
    if (first_line(out_abs) == "# " + header) return false;

    generate(header, p, table_abs, out_abs);
    std::printf("[gait] regenerated %s (stiffness %.3g) -> %s\n",
                label.c_str(), p.stiffness, activation_gait_path(label).c_str());
    return true;
}
