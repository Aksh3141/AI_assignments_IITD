#include "solver.h"
#include <iostream>
#include <chrono>
#include <random>
using namespace std;
#include <climits>
#include <unordered_map>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <limits>
#include <iomanip>
#include "structures.h"
#include <algorithm> 
#include<numeric>
// Result struct returned by the checker
struct CheckResult {
    bool valid;                      // true if no constraint violations found
    vector<string> errors;           // human readable list of violations
    double total_value = 0.0;        // total value achieved minus trip cost
    double total_trip_cost = 0.0;    // total cost across all trips
    // per-helicopter total distance traveled
    unordered_map<int,double> heli_total_distance;
    // per-village delivered counts {dry, perishable, other}
    unordered_map<int, array<int,3>> village_delivered;
};

// static string fmt_double(double x, int prec = 6) {
//     ostringstream oss;
//     oss << fixed << setprecision(prec) << x;
//     return oss.str();
// }

CheckResult checkSolution(const ProblemData& problem, const Solution& sol) {
    CheckResult res;
    res.valid = true;
    res.total_value = 0.0;
    res.total_trip_cost = 0.0;

    unordered_map<int, Helicopter> heli_map;
    for (auto& h : problem.helicopters) heli_map[h.id] = h;

    // Track per-village deliveries (needed for incremental capping)
    vector<double> food_delivered(problem.villages.size() + 1, 0.0);
    vector<double> other_delivered(problem.villages.size() + 1, 0.0);
    vector<double> village_values(problem.villages.size() + 1, 0.0);

    unordered_map<int,int> village_index;
    for (size_t i=0; i<problem.villages.size(); i++)
        village_index[problem.villages[i].id] = i;

    for (auto& plan : sol) {
        if (!heli_map.count(plan.helicopter_id)) continue;
        const Helicopter &H = heli_map[plan.helicopter_id];
        res.heli_total_distance[H.id] = 0.0;

        Point home = problem.cities[H.home_city_id-1];

        for (auto& trip : plan.trips) {
            // Weight check
            double w = trip.dry_food_pickup*problem.packages[0].weight +
                       trip.perishable_food_pickup*problem.packages[1].weight +
                       trip.other_supplies_pickup*problem.packages[2].weight;
            if (w > H.weight_capacity + 1e-9) res.valid = false;

            Point cur = home;
            double dist = 0.0;
            int dropped_d=0,dropped_p=0,dropped_o=0;

            for (auto& d : trip.drops) {
                if (!village_index.count(d.village_id)) continue;
                const Village& v = problem.villages[village_index[d.village_id]];

                // Distance
                dist += distance(cur, v.coords);
                cur = v.coords;

                // Value capping logic
                double max_food_needed = v.population * 9.0;
                double food_room_left = max(0.0, max_food_needed - food_delivered[d.village_id]);
                double food_in_this_drop = d.dry_food + d.perishable_food;
                double effective_food_this_drop = min(food_in_this_drop, food_room_left);

                double effective_vp = min((double)d.perishable_food, effective_food_this_drop);
                double value_from_p = effective_vp * problem.packages[1].value;

                double remaining_effective_food = effective_food_this_drop - effective_vp;
                double effective_vd = min((double)d.dry_food, remaining_effective_food);
                double value_from_d = effective_vd * problem.packages[0].value;

                double max_other_needed = v.population * 1.0;
                double other_room_left = max(0.0, max_other_needed - other_delivered[d.village_id]);
                double effective_vo = min((double)d.other_supplies, other_room_left);
                double value_from_o = effective_vo * problem.packages[2].value;

                village_values[d.village_id] += value_from_p + value_from_d + value_from_o;

                food_delivered[d.village_id] += food_in_this_drop;
                other_delivered[d.village_id] += d.other_supplies;

                dropped_d += d.dry_food;
                dropped_p += d.perishable_food;
                dropped_o += d.other_supplies;
            }

            // Check not dropping more than picked
            if (dropped_d > trip.dry_food_pickup ||
                dropped_p > trip.perishable_food_pickup ||
                dropped_o > trip.other_supplies_pickup)
                res.valid = false;

            dist += distance(cur, home);
            if (dist > H.distance_capacity + 1e-9) res.valid = false;

            res.heli_total_distance[H.id] += dist;
            if (!trip.drops.empty())
                res.total_trip_cost += H.fixed_cost + H.alpha*dist;
        }

        if (res.heli_total_distance[H.id] > problem.d_max + 1e-9)
            res.valid = false;
    }

    double total_value = accumulate(village_values.begin(), village_values.end(), 0.0);
    res.total_value = total_value - res.total_trip_cost;

    if (!res.valid) res.total_value = -1.0; // match file-checker
    return res;
}


int randInt(int lo, int hi, mt19937 &rng) {
    uniform_int_distribution<int> dist(lo, hi);
    return dist(rng);
}
double randDouble(double lo, double hi, mt19937 &rng) {
    uniform_real_distribution<double> dist(lo, hi);
    return dist(rng);
}
inline bool valid_city_index(int idx, const vector<Point> &cities) {
    return idx >= 0 && (size_t)idx < cities.size();
}
static int rndInt(int lo, int hi, std::mt19937 &rng) {
    if (hi <= lo) return lo;
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng);
}

// Helper: find helicopter in ProblemData by id
static const Helicopter* findHeliById(const ProblemData &problem, int heli_id) {
    for (const auto &h : problem.helicopters) if (h.id == heli_id) return &h;
    return nullptr;
}

/*
 * --- Neighbourhood move implementations ---
 * Each move modifies the passed-in `sol` in-place and
 * returns true if it made a change (so caller can evaluate).
 */

// Move A: transfer small amount of some package from one drop to another within same trip
static bool move_reassign_within_trip(const ProblemData &problem, Solution &sol, std::mt19937 &rng) {
    if (sol.empty()) return false;
    int pidx = rndInt(0, (int)sol.size()-1, rng);
    auto &plan = sol[pidx];
    if (plan.trips.empty()) return false;
    int tidx = rndInt(0, (int)plan.trips.size()-1, rng);
    auto &trip = plan.trips[tidx];
    if (trip.drops.size() < 2) return false;

    // pick source drop that has >0 of any package
    std::vector<int> srcs;
    for (int i=0;i<(int)trip.drops.size();++i) {
        const auto &d = trip.drops[i];
        if (d.perishable_food + d.dry_food + d.other_supplies > 0) srcs.push_back(i);
    }
    if (srcs.empty()) return false;
    int s = srcs[rndInt(0,(int)srcs.size()-1,rng)];

    // pick dest
    std::vector<int> dests;
    for (int i=0;i<(int)trip.drops.size();++i) if (i!=s) dests.push_back(i);
    if (dests.empty()) return false;
    int u = dests[rndInt(0,(int)dests.size()-1,rng)];

    // choose package type from source that is available
    std::vector<int> types;
    if (trip.drops[s].perishable_food > 0) types.push_back(0); // 0 -> perishable
    if (trip.drops[s].dry_food > 0) types.push_back(1);       // 1 -> dry
    if (trip.drops[s].other_supplies > 0) types.push_back(2); // 2 -> other
    if (types.empty()) return false;

    int typ = types[rndInt(0,(int)types.size()-1,rng)];
    int avail = (typ==0) ? trip.drops[s].perishable_food : (typ==1) ? trip.drops[s].dry_food : trip.drops[s].other_supplies;
    int amt = rndInt(1, std::min(3, avail), rng); // transfer small chunk (1..3)
    if (amt <= 0) return false;

    // perform transfer
    if (typ==0) { trip.drops[s].perishable_food -= amt; trip.drops[u].perishable_food += amt; }
    else if (typ==1) { trip.drops[s].dry_food -= amt; trip.drops[u].dry_food += amt; }
    else { trip.drops[s].other_supplies -= amt; trip.drops[u].other_supplies += amt; }

    return true;
}

// Move B: move a drop from one trip to another trip (maybe different helicopter)
static bool move_move_drop_between_trips(const ProblemData &problem, Solution &sol, std::mt19937 &rng) {
    // Collect all (plan,trip,drop) positions
    struct Pos { int p,t,d; };
    std::vector<Pos> positions;
    for (int pi=0; pi<(int)sol.size(); ++pi) {
        const auto &plan = sol[pi];
        for (int ti=0; ti<(int)plan.trips.size(); ++ti) {
            const auto &trip = plan.trips[ti];
            for (int di=0; di<(int)trip.drops.size(); ++di) {
                // only consider non-zero drops
                const auto &dr = trip.drops[di];
                if (dr.perishable_food + dr.dry_food + dr.other_supplies > 0)
                    positions.push_back({pi,ti,di});
            }
        }
    }
    if (positions.empty()) return false;
    // pick source
    Pos source = positions[rndInt(0,(int)positions.size()-1,rng)];

    // pick a target trip (different trip ideally)
    std::vector<std::pair<int,int>> trip_positions; // (plan_idx, trip_idx)
    for (int pi=0; pi<(int)sol.size(); ++pi) {
        for (int ti=0; ti<(int)sol[pi].trips.size(); ++ti) {
            if (pi==source.p && ti==source.t) continue;
            trip_positions.emplace_back(pi,ti);
        }
    }
    if (trip_positions.empty()) return false;
    auto target_pair = trip_positions[rndInt(0,(int)trip_positions.size()-1,rng)];
    int tpi = target_pair.first, tti = target_pair.second;

    // perform move: remove drop from source, append/merge into target trip
    Drop moved = sol[source.p].trips[source.t].drops[source.d];
    // erase source drop
    sol[source.p].trips[source.t].drops.erase(sol[source.p].trips[source.t].drops.begin() + source.d);

    // try merge into existing drop in target with same village
    bool merged = false;
    for (auto &dd : sol[tpi].trips[tti].drops) {
        if (dd.village_id == moved.village_id) {
            dd.dry_food += moved.dry_food;
            dd.perishable_food += moved.perishable_food;
            dd.other_supplies += moved.other_supplies;
            merged = true;
            break;
        }
    }
    if (!merged) {
        sol[tpi].trips[tti].drops.push_back(moved);
    }
    return true;
}

// Move C: swap two drops from two different trips (swap their content & village)
static bool move_swap_drops_between_trips(const ProblemData &problem, Solution &sol, std::mt19937 &rng) {
    // collect positions same as above
    struct Pos2 { int p,t,d; };
    std::vector<Pos2> pos;
    for (int pi=0; pi<(int)sol.size(); ++pi) {
        for (int ti=0; ti<(int)sol[pi].trips.size(); ++ti) {
            for (int di=0; di<(int)sol[pi].trips[ti].drops.size(); ++di) {
                pos.push_back({pi,ti,di});
            }
        }
    }
    if (pos.size() < 2) return false;
    int a = rndInt(0,(int)pos.size()-1,rng);
    int b = rndInt(0,(int)pos.size()-1,rng);
    if (a==b) return false;
    auto A = pos[a], B = pos[b];
    // swap Drops
    std::swap(sol[A.p].trips[A.t].drops[A.d], sol[B.p].trips[B.t].drops[B.d]);
    return true;
}

// Move D: increase pickups slightly (if weight allows) and assign those extra units to some drop in the trip
static bool move_increase_pickup_and_assign(const ProblemData &problem, Solution &sol, std::mt19937 &rng) {
    if (sol.empty()) return false;
    int pidx = rndInt(0, (int)sol.size()-1, rng);
    auto &plan = sol[pidx];
    if (plan.trips.empty()) return false;
    int tidx = rndInt(0, (int)plan.trips.size()-1, rng);
    auto &trip = plan.trips[tidx];

    const Helicopter *h = findHeliById(problem, plan.helicopter_id);
    if (!h) return false;

    double w_d = problem.packages[0].weight;
    double w_p = problem.packages[1].weight;
    double w_o = problem.packages[2].weight;

    double current_w = trip.dry_food_pickup * w_d + trip.perishable_food_pickup * w_p + trip.other_supplies_pickup * w_o;
    double remain_w = h->weight_capacity - current_w;
    if (remain_w <= 0.0) return false;

    // compute max additions for each type
    int max_add_p = (w_p>0.0) ? (int)floor(remain_w / w_p) : 0;
    int max_add_d = (w_d>0.0) ? (int)floor(remain_w / w_d) : 0;
    int max_add_o = (w_o>0.0) ? (int)floor(remain_w / w_o) : 0;
    std::vector<int> avail_types;
    if (max_add_p > 0) avail_types.push_back(0);
    if (max_add_d > 0) avail_types.push_back(1);
    if (max_add_o > 0) avail_types.push_back(2);
    if (avail_types.empty()) return false;

    int typ = avail_types[rndInt(0,(int)avail_types.size()-1,rng)];
    int cap = (typ==0) ? max_add_p : (typ==1) ? max_add_d : max_add_o;
    int delta = rndInt(1, std::min(3, cap), rng);

    if (typ==0) trip.perishable_food_pickup += delta;
    else if (typ==1) trip.dry_food_pickup += delta;
    else trip.other_supplies_pickup += delta;

    // assign these extra units to an existing drop (prefer) or create new drop at random
    if (!trip.drops.empty()) {
        int di = rndInt(0,(int)trip.drops.size()-1,rng);
        if (typ==0) trip.drops[di].perishable_food += delta;
        else if (typ==1) trip.drops[di].dry_food += delta;
        else trip.drops[di].other_supplies += delta;
    } else {
        // create a drop at a random village (use actual village id)
        int vidx = rndInt(0, (int)problem.villages.size()-1, rng);
        Drop dr;
        dr.village_id = problem.villages[vidx].id;
        dr.dry_food = (typ==1) ? delta : 0;
        dr.perishable_food = (typ==0) ? delta : 0;
        dr.other_supplies = (typ==2) ? delta : 0;
        trip.drops.push_back(dr);
    }
    return true;
}

/*
 * --- Greedy local search that samples neighbours and accepts the best-improving one ---
 *
 * Parameters:
 *  - problem: problem instance
 *  - init: initial solution to improve (copied)
 *  - max_iters: maximum outer iterations (stop earlier if no improvement)
 *  - neighbors_per_iter: how many random neighbours to sample per outer iteration
 *
 * Returns improved solution (or original if no improvement)
 */

Solution improveSolutionGreedy(
    const ProblemData &problem,Solution init,int max_iters = 1000,int neighbors_per_iter = 100){
    Solution best = init;
    std::mt19937 rng(std::random_device{}());

    CheckResult base_cr = checkSolution(problem, best);
    double best_score = base_cr.total_value;
    if (best_score <= -1e17) best_score = -1e18; // safety

    std::cout << "Starting greedy local improvement: initial score = " << base_cr.total_value << "\n";

    for (int it = 0; it < max_iters; ++it) {
        double best_neighbor_score = best_score;
        Solution best_neighbor;
        bool found_improvement = false;

        for (int k = 0; k < neighbors_per_iter; ++k) {
            Solution cand = best; // copy
            bool changed = false;
            int move_type = rndInt(0, 3, rng);
            switch (move_type) {
                case 0: changed = move_reassign_within_trip(problem, cand, rng); break;
                case 1: changed = move_move_drop_between_trips(problem, cand, rng); break;
                case 2: changed = move_swap_drops_between_trips(problem, cand, rng); break;
                case 3: changed = move_increase_pickup_and_assign(problem, cand, rng); break;
            }
            if (!changed) continue;

            CheckResult cr = checkSolution(problem, cand);
            // valid solutions will have total_value != -1.0 (assuming checker sets -1 on invalid)
            if (!cr.valid) continue;
            double score = cr.total_value;
            if (score > best_neighbor_score) {
                best_neighbor_score = score;
                best_neighbor = std::move(cand);
                found_improvement = true;
            }
        } // neighbor sampling

        if (!found_improvement) {
            std::cout << "No improving neighbour found at iteration " << it << " — stopping.\n";
            break;
        }
        // accept best neighbour
        best = std::move(best_neighbor);
        best_score = best_neighbor_score;
        //std::cout << "ITER " << it << " -> improved score = " << best_score << "\n";
    }

    std::cout << "Greedy improvement finished. best score = " << best_score << "\n";
    return best;
}

// Requires: #include <random>, <algorithm>, <numeric>, <climits>, "structures.h"
Solution randomSolutionGreedyRandom(const ProblemData &problem) {
    Solution sol;
    std::random_device rd;
    std::mt19937 rng(rd());

    if (problem.packages.size() < 3 || problem.villages.empty()) {
        for (const auto &h : problem.helicopters) {
            HelicopterPlan plan;
            plan.helicopter_id = h.id;
            sol.push_back(plan);
        }
        return sol;
    }

    // package params
    const double w_d = problem.packages[0].weight;
    const double v_d = problem.packages[0].value;
    const double w_p = problem.packages[1].weight;
    const double v_p = problem.packages[1].value;
    const double w_o = problem.packages[2].weight;
    const double v_o = problem.packages[2].value;

    const size_t V = problem.villages.size();

    // per-village remaining demand
    std::vector<int> food_remaining(V), other_remaining(V);
    for (size_t i = 0; i < V; ++i) {
        food_remaining[i]  = problem.villages[i].population * 9;   // perishable + dry
        other_remaining[i] = problem.villages[i].population * 1;
    }

    // helper: weighted choice via discrete_distribution
    auto choose_index_by_weights = [&](const std::vector<double>& weights)->int {
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (sum <= 0.0) return -1;
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        return dist(rng);
    };

    for (const auto &h : problem.helicopters) {
        HelicopterPlan plan;
        plan.helicopter_id = h.id;

        // home city index: file-checker assumed 1-based ids
        int home_idx = h.home_city_id - 1;
        if (home_idx < 0 || (size_t)home_idx >= problem.cities.size()) home_idx = 0;
        Point home = problem.cities[home_idx];

        double remaining_heli_distance = problem.d_max;
        int max_trips = std::max(1, (int)std::floor(problem.d_max / std::max(1.0, h.distance_capacity)));

        for (int t = 0; t < max_trips; ++t) {
            if (remaining_heli_distance <= 1e-9) break;

            long long total_food_left = std::accumulate(food_remaining.begin(), food_remaining.end(), 0LL);
            long long total_other_left = std::accumulate(other_remaining.begin(), other_remaining.end(), 0LL);
            if (total_food_left <= 0 && total_other_left <= 0) break;

            Trip trip;
            trip.dry_food_pickup = 0;
            trip.perishable_food_pickup = 0;
            trip.other_supplies_pickup = 0;

            double remaining_w = h.weight_capacity;

            // Decide pickups biased by value/weight ratio (greedy)
            struct Item { double ratio; int type; }; // 0:perishable,1:dry,2:other
            std::vector<Item> items = { {v_p / (w_p+1e-12), 0}, {v_d / (w_d+1e-12), 1}, {v_o / (w_o+1e-12), 2} };
            std::sort(items.begin(), items.end(), [](const Item&a,const Item&b){ return a.ratio > b.ratio; });

            // pick for each package type in ratio order using uniform_int_distribution<int>
            for (auto &it : items) {
                if (remaining_w <= 1e-9) break;
                if (it.type == 0) { // perishable
                    int can = (w_p > 0.0) ? (int)std::floor(remaining_w / w_p) : 0;
                    int global_food = (int)std::min<long long>(total_food_left, (long long)INT_MAX);
                    int max_take = std::min(can, global_food);
                    if (max_take <= 0) continue;
                    std::uniform_int_distribution<int> dist(0, max_take);
                    int take = dist(rng);
                    trip.perishable_food_pickup = take;
                    remaining_w -= (double)take * w_p;
                    total_food_left = std::max(0LL, total_food_left - take);
                } else if (it.type == 1) { // dry
                    int can = (w_d > 0.0) ? (int)std::floor(remaining_w / w_d) : 0;
                    int global_food = (int)std::min<long long>(total_food_left, (long long)INT_MAX);
                    int max_take = std::min(can, global_food);
                    if (max_take <= 0) continue;
                    std::uniform_int_distribution<int> dist(0, max_take);
                    int take = dist(rng);
                    trip.dry_food_pickup = take;
                    remaining_w -= (double)take * w_d;
                    total_food_left = std::max(0LL, total_food_left - take);
                } else { // other
                    int can = (w_o > 0.0) ? (int)std::floor(remaining_w / w_o) : 0;
                    int global_other = (int)std::min<long long>(total_other_left, (long long)INT_MAX);
                    int max_take = std::min(can, global_other);
                    if (max_take <= 0) continue;
                    std::uniform_int_distribution<int> dist(0, max_take);
                    int take = dist(rng);
                    trip.other_supplies_pickup = take;
                    remaining_w -= (double)take * w_o;
                    total_other_left = std::max(0LL, total_other_left - take);
                }
            }

            int total_picked = trip.perishable_food_pickup + trip.dry_food_pickup + trip.other_supplies_pickup;
            if (total_picked == 0) continue;

            // prepare candidate villages (weights prefer large population and near home)
            std::vector<double> weights(V, 0.0);
            for (size_t i = 0; i < V; ++i) {
                if (food_remaining[i] <= 0 && other_remaining[i] <= 0) continue;
                double d = distance(home, problem.villages[i].coords);
                double score = (double)problem.villages[i].population / (1.0 + d); // higher = better
                weights[i] = std::max(0.0, score);
            }

            // allocate trip drops greedily but with randomness (weighted village choice)
            int rem_p = trip.perishable_food_pickup;
            int rem_d = trip.dry_food_pickup;
            int rem_o = trip.other_supplies_pickup;

            Point cur = home;
            double trip_dist = 0.0;
            std::vector<Drop> drops_for_trip;

            int attempt_guard = 0;
            while ((rem_p + rem_d + rem_o) > 0 && attempt_guard < (int)V * 4) {
                ++attempt_guard;
                int vidx = choose_index_by_weights(weights);
                if (vidx < 0) break;

                Point vpt = problem.villages[vidx].coords;
                double dist_cur_home = distance(cur, home);
                double dist_cur_v = distance(cur, vpt);
                double dist_v_home = distance(vpt, home);
                double new_trip_dist = trip_dist - dist_cur_home + dist_cur_v + dist_v_home;

                if (new_trip_dist > h.distance_capacity + 1e-9 || new_trip_dist > remaining_heli_distance + 1e-9) {
                    weights[vidx] = 0.0;
                    continue;
                }

                int give_p = 0, give_d = 0, give_o = 0;
                int food_room = food_remaining[vidx];

                if (food_room > 0 && rem_p > 0) {
                    give_p = std::min(rem_p, food_room);
                    food_room -= give_p;
                }
                if (food_room > 0 && rem_d > 0) {
                    give_d = std::min(rem_d, food_room);
                    food_room -= give_d;
                }
                if (other_remaining[vidx] > 0 && rem_o > 0) {
                    give_o = std::min(rem_o, other_remaining[vidx]);
                }

                if (give_p + give_d + give_o == 0) {
                    weights[vidx] = 0.0;
                    continue;
                }

                Drop dr;
                dr.village_id = problem.villages[vidx].id;
                dr.dry_food = give_d;
                dr.perishable_food = give_p;
                dr.other_supplies = give_o;

                drops_for_trip.push_back(dr);

                rem_p -= give_p; rem_d -= give_d; rem_o -= give_o;
                food_remaining[vidx] -= (give_p + give_d);
                other_remaining[vidx] -= give_o;

                trip_dist = new_trip_dist;
                cur = vpt;

                if (food_remaining[vidx] <= 0 && other_remaining[vidx] <= 0) weights[vidx] = 0.0;
            }

            if (drops_for_trip.empty()) continue;

            trip.drops = std::move(drops_for_trip);

            if (trip_dist > h.distance_capacity + 1e-9) continue;
            if (trip_dist > remaining_heli_distance + 1e-9) continue;

            remaining_heli_distance -= trip_dist;
            plan.trips.push_back(std::move(trip));
        } // trips loop

        sol.push_back(std::move(plan));
    } // helis loop

    return sol;
}



// Solution solve(const ProblemData& problem) {
//     cout << "Starting random search solver..." << endl;
//     Solution best_solution;
//     double best_value = -1e18;
//     int num_samples = 1000000;
//     for (int i = 0; i < num_samples; ++i) {
//         Solution candidate = randomSolution(problem);
//         CheckResult cr = checkSolution(problem, candidate);
//         if (cr.total_value > best_value) {
//             best_value = cr.total_value;
//             best_solution = candidate;
//         }
//     }
//     if (best_solution.empty()) {
//         cout << "No valid solution found after " << num_samples << " samples.\n";
//     } else {
//         CheckResult cr = checkSolution(problem, best_solution);
//         cout << "Best solution found with value = " << cr.total_value
//              << ", total cost = " << cr.total_trip_cost << endl;
//         for (auto &p : cr.heli_total_distance) {
//             cout << "Heli " << p.first << " total distance = " << p.second << "\n";
//         }
//     }
//     cout << "Solver finished." << endl;
//     return best_solution;
// }


// Solution solve(const ProblemData& problem) {
//     cout << "Starting random search solver..." << endl;
//     Solution best_solution;
//     double best_value = -1e18;
//     // Record start time
//     auto start = chrono::high_resolution_clock::now();
//     auto time_limit = chrono::duration<double>((problem.time_limit_minutes * 60.0) - 10.0);
//     while (true) {
//         auto now = chrono::high_resolution_clock::now();
//         if (now - start >= time_limit) break;
//         Solution candidate = randomSolution(problem);
//         CheckResult cr = checkSolution(problem, candidate);
//         if (cr.total_value > best_value) {
//             best_value = cr.total_value;
//             best_solution = candidate;
//         }
//     }
//     if (best_solution.empty()) {
//         cout << "No valid solution found.\n";
//     } else {
//         CheckResult cr = checkSolution(problem, best_solution);
//         cout << "\n--- Final Calculation ---" << endl;
//         cout << "Total Value Gained: " 
//              << (cr.total_value + cr.total_trip_cost) << endl;
//         cout << "Total Trip Cost   : " << cr.total_trip_cost << endl;
//         cout << "Objective Score   = " 
//              << (cr.total_value + cr.total_trip_cost)
//              << " - " << cr.total_trip_cost
//              << " = " << cr.total_value << endl;
//         if (!cr.valid) {
//             cout << "\n*** WARNING: CONSTRAINTS VIOLATED. Score is invalid. ***" << endl;
//         } else {
//             cout << "\n--- All constraints satisfied. ---" << endl;
//         }
//         cout << "\n----------------------------------------\n"
//              << "FINAL SCORE: " << cr.total_value
//              << "\n----------------------------------------" << endl;
//         for (auto &p : cr.heli_total_distance) {
//             cout << "Heli " << p.first << " total distance = " << p.second << "\n";
//         }
//     }
//     cout << "Solver finished" << endl;
//     return best_solution;
// }


Solution solve(const ProblemData &problem) {
    std::random_device rd;
    std::mt19937 rng(rd());

    Solution best_solution;
    double best_score = -1e18;

    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    auto time_limit = std::chrono::seconds(50);

    int it = 0;
    while (clock::now() - start < time_limit) {
        it++;

        // Step 1: generate random solution
        Solution candidate = randomSolutionGreedyRandom(problem);
        CheckResult cr = checkSolution(problem, candidate);

        if (!cr.valid) continue;  // skip invalid

        double score = cr.total_value;
        if (score < -1e17) continue;
        if (score > best_score) {
            // try greedy improvement only now
            candidate = improveSolutionGreedy(problem, candidate, /*max_iters*/ 100, /*neighbors*/ 50);
            cr = checkSolution(problem, candidate);
            if (!cr.valid) continue;

            score = cr.total_value;

            // Step 3: update global best if improved
            if (score > best_score) {
                best_score = score;
                best_solution = candidate;
                // std::cout << "NEW GLOBAL BEST (iter " << it << "): " << best_score << "\n";
            }
        }
    }

    std::cout << "Final Best Score = " << best_score << "\n";
    return best_solution;
}


