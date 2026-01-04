#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
using namespace std;

// ==============================
// Metro Problem Definition
// ==============================
struct MetroProblem {
    int scenario;
    int N, M, K, J, P;
    vector<pair<int,int>> starts;
    vector<pair<int,int>> ends;
};

// ==============================
// Variable Mapping
// ==============================
struct MetroVarMap {
    int N, M, K, J;
    int off_p, off_h, off_v, off_turn, off_s;
    int total;

    MetroVarMap(const MetroProblem& pr) {
        N = pr.N; M = pr.M; K = pr.K; J = pr.J;
        int cells = M * N;
        int np = K * cells;
        int nh = K * M * (N - 1);
        int nv = K * (M - 1) * N;
        int nt = K * cells;
        int ns = K * (cells + 1) * (J + 2);

        off_p = 1;
        off_h = off_p + np;
        off_v = off_h + nh;
        off_turn = off_v + nv;
        off_s = off_turn + nt;
        total = np + nh + nv + nt + ns;
    }

    int p(int k, int r, int c) const { return off_p + k * (M * N) + r * N + c; }
    int h(int k, int r, int c) const { return off_h + k * (M * (N - 1)) + r * (N - 1) + c; }
    int v(int k, int r, int c) const { return off_v + k * ((M - 1) * N) + r * N + c; }
    int turn(int k, int r, int c) const { return off_turn + k * (M * N) + r * N + c; }
    int s(int k, int cell_idx, int j) const { return off_s + k * ((M * N + 1) * (J + 2)) + cell_idx * (J + 2) + j; }
};

// ==============================
// Read City File
// ==============================
void read_city(const string& name, MetroProblem& pr) {
    ifstream f(name);
    if (!f) throw runtime_error("Cannot open city file: " + name);

    f >> pr.scenario;
    if (pr.scenario == 1) f >> pr.N >> pr.M >> pr.K >> pr.J, pr.P = 0;
    else f >> pr.N >> pr.M >> pr.K >> pr.J >> pr.P;

    pr.starts.resize(pr.K);
    pr.ends.resize(pr.K);
    for (int k = 0; k < pr.K; ++k) {
        f >> pr.starts[k].second >> pr.starts[k].first;
        f >> pr.ends[k].second >> pr.ends[k].first;
    }
}

// ==============================
// Main
// ==============================
int main(int argc, char* argv[]) {

    if (argc != 3) {
        cerr << "Usage: ./metro_decoder <city_file> <sat_file>\n";
        return 1;
    }

    string city_file = argv[1], sat_file = argv[2];
    ifstream sat_in(sat_file);
    if (!sat_in) {
        cerr << "Cannot open SAT file: " << sat_file << "\n";
        return 1;
    }

    string result;
    sat_in >> result;
    if (result == "UNSAT") { cout << "0\n"; return 0; }
    if (result != "SAT") {
        cerr << "SAT file must start with SAT or UNSAT\n";
        return 1;
    }

    MetroProblem pr;
    try { read_city(city_file, pr); }
    catch (const exception& e) { cerr << e.what() << "\n"; return 1; }

    MetroVarMap vars(pr);

    vector<bool> val(vars.total + 1, false);
    int lit;
    while (sat_in >> lit && lit != 0) if (lit > 0 && lit <= vars.total) val[lit] = true;

    for (int k = 0; k < pr.K; ++k) {
        auto now = pr.starts[k];
        auto goal = pr.ends[k];
        auto prev = make_pair(-1, -1);
        string path;

        while (now != goal) {
            int r = now.first, c = now.second;
            bool moved = false;

            if (c + 1 < pr.N && make_pair(r, c + 1) != prev && val[vars.h(k, r, c)]) {
                path += 'R'; prev = now; now = {r, c + 1}; moved = true;
            }
            else if (c - 1 >= 0 && make_pair(r, c - 1) != prev && val[vars.h(k, r, c - 1)]) {
                path += 'L'; prev = now; now = {r, c - 1}; moved = true;
            }
            else if (r + 1 < pr.M && make_pair(r + 1, c) != prev && val[vars.v(k, r, c)]) {
                path += 'D'; prev = now; now = {r + 1, c}; moved = true;
            }
            else if (r - 1 >= 0 && make_pair(r - 1, c) != prev && val[vars.v(k, r - 1, c)]) {
                path += 'U'; prev = now; now = {r - 1, c}; moved = true;
            }

            if (!moved) {
                cerr << "Error: stuck metro " << k << " at (" << r << "," << c << ")\n";
                break;
            }
        }

        // ---- Print with spaces between characters ----
        for (char ch : path) cout << ch << ' ';
        cout << '0' << "\n";
    }

    return 0;
}