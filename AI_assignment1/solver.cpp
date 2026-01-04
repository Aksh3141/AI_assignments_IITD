using namespace std;
#include "solver.h"
#include<bits/stdc++.h> 

// Result struct returned by the checker
struct CheckResult {
    bool valid;               
    vector<string> errors;           
    double total_value = 0.0;        
    double total_trip_cost = 0.0;    
    // per-helicopter total distance traveled
    unordered_map<int,double> heli_total_distance;
    // per-village delivered counts {dry, perishable, other}
    unordered_map<int, array<int,3>> village_delivered;
};

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

                // Value capping 
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

    if (!res.valid) res.total_value = -1.0; 
    return res;
}


static int rndInt(int lo, int hi, mt19937 &rng) {
    if (hi <= lo) return lo;
    uniform_int_distribution<int> dist(lo, hi);
    return dist(rng);
}

// find helicopter in ProblemData by id
static const Helicopter* findHeliById(const ProblemData &problem, int heli_id) {
    for (const auto &h : problem.helicopters) if (h.id == heli_id) return &h;
    return nullptr;
}

// compute trip distance given home and drop villages (order matters)
static double compute_trip_distance(const Point &home, const vector<Drop> &drops, const ProblemData &problem) {
    double d = 0.0;
    Point cur = home;
    unordered_map<int,int> vindex;
    for (size_t i=0;i<problem.villages.size();++i) vindex[problem.villages[i].id] = i;
    for (const auto &dr : drops) {
        if (!vindex.count(dr.village_id)) continue;
        Point vpt = problem.villages[vindex[dr.village_id]].coords;
        d += distance(cur, vpt);
        cur = vpt;
    }
    d += distance(cur, home);
    return d;
}

// drop ordering to reduce distance (modifies drops)
static bool two_opt_improve_trip_order(const Point &home, vector<Drop> &drops, const ProblemData &problem) {
    if (drops.size() < 3) return false;
    bool improved = false;
    // build coords vector
    unordered_map<int,int> vindex;
    for (size_t i=0;i<problem.villages.size();++i) vindex[problem.villages[i].id] = i;
    double best = compute_trip_distance(home, drops, problem);
    for (size_t i=0;i<drops.size()-1;++i) {
        for (size_t k=i+1;k<drops.size();++k) {
            vector<Drop> cand = drops;
            reverse(cand.begin()+i, cand.begin()+k+1);
            double d = compute_trip_distance(home, cand, problem);
            if (d + 1e-9 < best) {
                best = d;
                drops = move(cand);
                improved = true;
                return true; 
            }
        }
    }
    return improved;
}

// ----------------------- Moves to neighbours of a state -----------------------

static bool move_reassign_within_trip(const ProblemData &problem, Solution &sol, mt19937 &rng) {
    if (sol.empty()) return false;
    int pidx = rndInt(0, (int)sol.size()-1, rng);
    auto &plan = sol[pidx];
    if (plan.trips.empty()) return false;
    int tidx = rndInt(0, (int)plan.trips.size()-1, rng);
    auto &trip = plan.trips[tidx];
    if (trip.drops.size() < 2) return false;

    // pick source drop that has >0 of any package
    vector<int> srcs;
    for (int i=0;i<(int)trip.drops.size();++i) {
        const auto &d = trip.drops[i];
        if (d.perishable_food + d.dry_food + d.other_supplies > 0) srcs.push_back(i);
    }
    if (srcs.empty()) return false;
    int s = srcs[rndInt(0,(int)srcs.size()-1,rng)];

    // pick dest
    vector<int> dests;
    for (int i=0;i<(int)trip.drops.size();++i) if (i!=s) dests.push_back(i);
    if (dests.empty()) return false;
    int u = dests[rndInt(0,(int)dests.size()-1,rng)];

    // choose package type from source that is available
    vector<int> types;
    if (trip.drops[s].perishable_food > 0) types.push_back(0); // 0 -> perishable
    if (trip.drops[s].dry_food > 0) types.push_back(1);       // 1 -> dry
    if (trip.drops[s].other_supplies > 0) types.push_back(2); // 2 -> other
    if (types.empty()) return false;

    int typ = types[rndInt(0,(int)types.size()-1,rng)];
    int avail = (typ==0) ? trip.drops[s].perishable_food : (typ==1) ? trip.drops[s].dry_food : trip.drops[s].other_supplies;
    int amt = rndInt(1, min(5, avail), rng); // allow slightly larger chunks
    if (amt <= 0) return false;

    // perform transfer
    if (typ==0) { trip.drops[s].perishable_food -= amt; trip.drops[u].perishable_food += amt; }
    else if (typ==1) { trip.drops[s].dry_food -= amt; trip.drops[u].dry_food += amt; }
    else { trip.drops[s].other_supplies -= amt; trip.drops[u].other_supplies += amt; }

    return true;
}

static bool move_swap_drops_between_trips(const ProblemData &problem, Solution &sol, mt19937 &rng) {
    // collect positions same as above
    struct Pos2 { int p,t,d; };
    vector<Pos2> pos;
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
    swap(sol[A.p].trips[A.t].drops[A.d], sol[B.p].trips[B.t].drops[B.d]);
    return true;
}

static bool move_relocate_trip_between_helicopters(const ProblemData &problem, Solution &sol, mt19937 &rng) {
    if (sol.size() < 2) return false;
    int src_plan = rndInt(0, (int)sol.size()-1, rng);
    if (sol[src_plan].trips.empty()) return false;
    int trip_idx = rndInt(0, (int)sol[src_plan].trips.size()-1, rng);
    Trip moving = sol[src_plan].trips[trip_idx];

    // choose target plan different from source
    vector<int> candidates;
    for (int i=0;i<(int)sol.size();++i) if (i!=src_plan) candidates.push_back(i);
    if (candidates.empty()) return false;
    int tgt_plan = candidates[rndInt(0,(int)candidates.size()-1,rng)];

    const Helicopter *tgt_h = findHeliById(problem, sol[tgt_plan].helicopter_id);
    const Helicopter *src_h = findHeliById(problem, sol[src_plan].helicopter_id);
    if (!tgt_h || !src_h) return false;

    // quick feasibility check: weight
    double w_d = problem.packages[0].weight;
    double w_p = problem.packages[1].weight;
    double w_o = problem.packages[2].weight;
    double total_w = moving.dry_food_pickup*w_d + moving.perishable_food_pickup*w_p + moving.other_supplies_pickup*w_o;
    if (total_w > tgt_h->weight_capacity + 1e-9) return false;

    // check distance for target heli (approx): build home point
    Point tgt_home = problem.cities[tgt_h->home_city_id - 1];
    double trip_dist = compute_trip_distance(tgt_home, moving.drops, problem);
    if (trip_dist > tgt_h->distance_capacity + 1e-9) return false;

    // perform move
    sol[tgt_plan].trips.push_back(moving);
    // erase from source
    sol[src_plan].trips.erase(sol[src_plan].trips.begin() + trip_idx);
    return true;
}
static bool move_optimize_trip_route(const ProblemData &problem, Solution &sol,mt19937 &rng) {
    if (sol.empty()) return false;
    int pidx = rndInt(0, (int)sol.size()-1, rng);
    auto &plan = sol[pidx];
    if (plan.trips.empty()) return false;
    int tidx = rndInt(0, (int)plan.trips.size()-1, rng);
    auto &trip = plan.trips[tidx];
    const Helicopter *h = findHeliById(problem, plan.helicopter_id);
    if (!h) return false;
    Point home = problem.cities[h->home_city_id - 1];
    return two_opt_improve_trip_order(home, trip.drops, problem);
}
static bool move_relocate_drop_position(const ProblemData &problem, Solution &sol,mt19937 &rng) {
    if (sol.empty()) return false;
    int pidx = rndInt(0, (int)sol.size()-1, rng);
    auto &plan = sol[pidx];
    if (plan.trips.empty()) return false;
    int tidx = rndInt(0, (int)plan.trips.size()-1, rng);
    auto &trip = plan.trips[tidx];
    if (trip.drops.size() < 2) return false;
    int from = rndInt(0,(int)trip.drops.size()-1,rng);
    int to = rndInt(0,(int)trip.drops.size()-1,rng);
    if (from == to) return false;
    Drop tmp = trip.drops[from];
    trip.drops.erase(trip.drops.begin()+from);
    trip.drops.insert(trip.drops.begin()+min((int)trip.drops.size(), to), tmp);
    return true;
}
static bool move_relocate_drop_to_other_trip_same_heli(const ProblemData &problem, Solution &sol, mt19937 &rng) {
    // pick plan with >=2 trips
    vector<int> plans;
    for (int i=0;i<(int)sol.size();++i) if (sol[i].trips.size() >= 2) plans.push_back(i);
    if (plans.empty()) return false;
    int p = plans[rndInt(0,(int)plans.size()-1,rng)];
    int src_t = rndInt(0,(int)sol[p].trips.size()-1,rng);
    int dst_t = rndInt(0,(int)sol[p].trips.size()-1,rng);
    if (src_t == dst_t) return false;
    if (sol[p].trips[src_t].drops.empty()) return false;
    int drop_i = rndInt(0,(int)sol[p].trips[src_t].drops.size()-1,rng);
    Drop moved = sol[p].trips[src_t].drops[drop_i];
    sol[p].trips[src_t].drops.erase(sol[p].trips[src_t].drops.begin() + drop_i);
    // merge into dst if same village exists
    bool merged=false;
    for (auto &d : sol[p].trips[dst_t].drops) {
        if (d.village_id == moved.village_id) {
            d.dry_food += moved.dry_food;
            d.perishable_food += moved.perishable_food;
            d.other_supplies += moved.other_supplies;
            merged=true; break;
        }
    }
    if (!merged) sol[p].trips[dst_t].drops.push_back(moved);
    return true;
}
static bool move_rebalance_pickups_same_heli(const ProblemData &problem, Solution &sol, mt19937 &rng) {
    // pick a plan with >=2 trips
    vector<int> plans;
    for (int i=0;i<(int)sol.size();++i) if (sol[i].trips.size() >= 2) plans.push_back(i);
    if (plans.empty()) return false;

    int p = plans[rndInt(0,(int)plans.size()-1,rng)];
    auto &plan = sol[p];
    int t1 = rndInt(0,(int)plan.trips.size()-1,rng);
    int t2 = rndInt(0,(int)plan.trips.size()-1,rng);
    if (t1 == t2) return false;

    Trip &A = plan.trips[t1];
    Trip &B = plan.trips[t2];

    // choose type and direction
    vector<int> types;
    if (A.perishable_food_pickup > 0) types.push_back(0);
    if (A.dry_food_pickup > 0) types.push_back(1);
    if (A.other_supplies_pickup > 0) types.push_back(2);
    if (types.empty()) return false;

    int typ = types[rndInt(0,(int)types.size()-1,rng)];
    int cap = (typ==0) ? A.perishable_food_pickup : (typ==1) ? A.dry_food_pickup : A.other_supplies_pickup;
    if (cap <= 0) return false;

    int amt = rndInt(1, min(3, cap), rng);

    if (typ==0) { A.perishable_food_pickup -= amt; B.perishable_food_pickup += amt; }
    else if (typ==1) { A.dry_food_pickup -= amt; B.dry_food_pickup += amt; }
    else { A.other_supplies_pickup -= amt; B.other_supplies_pickup += amt; }

    return true;
}
static bool move_swap_pickups_between_helicopters(const ProblemData &problem, Solution &sol, mt19937 &rng) {
    if (sol.size() < 2) return false;

    int p1 = rndInt(0,(int)sol.size()-1,rng);
    int p2 = rndInt(0,(int)sol.size()-1,rng);
    if (p1 == p2) return false;

    auto &A = sol[p1];
    auto &B = sol[p2];
    if (A.trips.empty() || B.trips.empty()) return false;

    int t1 = rndInt(0,(int)A.trips.size()-1,rng);
    int t2 = rndInt(0,(int)B.trips.size()-1,rng);

    // swap one pickup field
    int typ = rndInt(0,2,rng);
    if (typ==0) swap(A.trips[t1].perishable_food_pickup, B.trips[t2].perishable_food_pickup);
    else if (typ==1) swap(A.trips[t1].dry_food_pickup, B.trips[t2].dry_food_pickup);
    else swap(A.trips[t1].other_supplies_pickup, B.trips[t2].other_supplies_pickup);

    return true;
}
static bool move_shuffle_trip_pickups(const ProblemData &problem, Solution &sol, mt19937 &rng) {
    if (sol.empty()) return false;
    int pidx = rndInt(0,(int)sol.size()-1,rng);
    if (sol[pidx].trips.empty()) return false;
    int tidx = rndInt(0,(int)sol[pidx].trips.size()-1,rng);
    auto &trip = sol[pidx].trips[tidx];
    if (trip.drops.size() < 2) return false;

    int typ = rndInt(0,2,rng);
    int total = 0;
    for (auto &d : trip.drops) {
        if (typ==0) { total += d.perishable_food; d.perishable_food = 0; }
        else if (typ==1) { total += d.dry_food; d.dry_food = 0; }
        else { total += d.other_supplies; d.other_supplies = 0; }
    }
    if (total <= 0) return false;

    // randomly distribute total back
    while (total > 0) {
        int give = min(total, rndInt(1,3,rng));
        int di = rndInt(0,(int)trip.drops.size()-1,rng);
        if (typ==0) trip.drops[di].perishable_food += give;
        else if (typ==1) trip.drops[di].dry_food += give;
        else trip.drops[di].other_supplies += give;
        total -= give;
    }
    return true;
}

// to check if time up 
inline bool timeExpired(
    const chrono::time_point<chrono::high_resolution_clock> &start,
    chrono::seconds limit)
{
    return chrono::high_resolution_clock::now() - start >= limit;
}

// ----------------------- Greedy random solution (more greedy pickups & village choice) -----------------------

Solution randomSolutionGreedyValid(
    const ProblemData &problem,chrono::time_point<chrono::high_resolution_clock> start,chrono::seconds time_limit) {
    
    Solution sol(problem.helicopters.size());
    if (problem.packages.size() < 3 || problem.villages.empty()) {
        for (const auto &h : problem.helicopters) {
            HelicopterPlan plan;
            plan.helicopter_id = h.id;
            sol.push_back(plan);
        }
        return sol;
    }

    // Package weights
    const double w_d = problem.packages[0].weight;
    const double w_p = problem.packages[1].weight;
    const double w_o = problem.packages[2].weight;

    const size_t V = problem.villages.size();

    // Remaining demand per village
    vector<int> food_remaining(V), other_remaining(V);
    for (size_t i = 0; i < V; ++i) {
        food_remaining[i] = problem.villages[i].population * 9;   // perishable + dry
        other_remaining[i] = problem.villages[i].population * 1;
    }

    random_device rd;
    mt19937 rng(rd());

    auto choose_index_by_weights = [&](const vector<double>& weights) -> int {
        double sum = accumulate(weights.begin(), weights.end(), 0.0);
        if (sum <= 0.0) return -1;
        discrete_distribution<int> dist(weights.begin(), weights.end());
        return dist(rng);
    };

    // Strategy choices
    auto fill_supplies = [&](Trip &trip, double &remaining_w,
                         long long total_food_left, long long total_other_left,
                         int strategy, mt19937 &rng) {
    auto take = [&](int &pickup, double w, long long avail) {
        int count = min((int)floor(remaining_w / w), (int)avail);
        pickup = count;
        remaining_w -= count * w;
    };

    if (strategy == 0) { // perishable -> other -> dry
        take(trip.perishable_food_pickup, w_p, total_food_left);
        take(trip.other_supplies_pickup, w_o, total_other_left);
        take(trip.dry_food_pickup, w_d, total_food_left);
    } else if (strategy == 1) { // dry -> perishable -> other
        take(trip.dry_food_pickup, w_d, total_food_left);
        take(trip.perishable_food_pickup, w_p, total_food_left);
        take(trip.other_supplies_pickup, w_o, total_other_left);
    } else if (strategy == 2) { 
        take(trip.other_supplies_pickup, w_o, total_other_left);
        take(trip.perishable_food_pickup, w_p, total_food_left);
        take(trip.dry_food_pickup, w_d, total_food_left);
    } else if (strategy == 3) { 
        vector<pair<char,double>> items = {{'p',w_p},{'d',w_d},{'o',w_o}};
        shuffle(items.begin(), items.end(), rng);
        for (auto &it : items) {
            if (it.first == 'p') take(trip.perishable_food_pickup, w_p, total_food_left);
            if (it.first == 'd') take(trip.dry_food_pickup, w_d, total_food_left);
            if (it.first == 'o') take(trip.other_supplies_pickup, w_o, total_other_left);
        }
    } else if (strategy == 4) { 
        vector<tuple<double,char>> items;
        items.push_back({problem.packages[0].value / w_d, 'd'});
        items.push_back({problem.packages[1].value / w_p, 'p'});
        items.push_back({problem.packages[2].value / w_o, 'o'});
        sort(items.begin(), items.end(), [](auto &a, auto &b) {
            return get<0>(a) > get<0>(b); 
        });
        for (auto &it : items) {
            if (get<1>(it) == 'p') take(trip.perishable_food_pickup, w_p, total_food_left);
            if (get<1>(it) == 'd') take(trip.dry_food_pickup, w_d, total_food_left);
            if (get<1>(it) == 'o') take(trip.other_supplies_pickup, w_o, total_other_left);
        }
    }
};

    // Generate plan per helicopter
    vector<int> heli_order((int)problem.helicopters.size());
    iota(heli_order.begin(), heli_order.end(), 0);   
    shuffle(heli_order.begin(), heli_order.end(), rng);
    for (int hi : heli_order) {
        if (timeExpired(start, time_limit)) {
        // fill empty plans for remaining helicopters
        for (int hj : heli_order) {
            if (sol[hj].helicopter_id == 0) { // not yet assigned
                HelicopterPlan emptyPlan;
                emptyPlan.helicopter_id = problem.helicopters[hj].id;
                sol[hj] = std::move(emptyPlan);
            }
        }
        return sol;
    }
        const auto &h = problem.helicopters[hi];
        HelicopterPlan plan;
        plan.helicopter_id = h.id;

        Point home = problem.cities[h.home_city_id - 1];
        double remaining_heli_distance = problem.d_max;

        while (remaining_heli_distance > 1e-9) {
            long long total_food_left = accumulate(food_remaining.begin(), food_remaining.end(), 0LL);
            long long total_other_left = accumulate(other_remaining.begin(), other_remaining.end(), 0LL);
            if (total_food_left <= 0 && total_other_left <= 0) break;

            Trip trip;
            trip.perishable_food_pickup = 0;
            trip.dry_food_pickup = 0;
            trip.other_supplies_pickup = 0;
            double remaining_w = h.weight_capacity;

            // --- randomized strategy choice ---
            uniform_int_distribution<int> strat_dist(0, 4);
            int strat_choice = strat_dist(rng);
            fill_supplies(trip, remaining_w, total_food_left, total_other_left, strat_choice, rng);

            if ((trip.perishable_food_pickup + trip.dry_food_pickup + trip.other_supplies_pickup) == 0)
                break;

            // Candidate villages weighted by population, distance, and unmet demand
            vector<double> weights(V, 0.0);
            for (size_t i = 0; i < V; ++i) {
                if (food_remaining[i] <= 0 && other_remaining[i] <= 0) continue;
                double d = distance(home, problem.villages[i].coords);
                double demand_score = food_remaining[i] + other_remaining[i];
                weights[i] = demand_score / (1.0 + d);
            }

            int rem_p = trip.perishable_food_pickup;
            int rem_d = trip.dry_food_pickup;
            int rem_o = trip.other_supplies_pickup;

            Point cur = home;
            double trip_dist = 0.0;
            vector<Drop> drops_for_trip;

            int attempt_guard = 0;
            while ((rem_p + rem_d + rem_o) > 0 && attempt_guard < (int)V * 4) {
                ++attempt_guard;
                int vidx = choose_index_by_weights(weights);
                if (vidx < 0) break;

                Point vpt = problem.villages[vidx].coords;
                double new_trip_dist = trip_dist - distance(cur, home) + distance(cur, vpt) + distance(vpt, home);

                if (new_trip_dist > h.distance_capacity + 1e-9 || new_trip_dist > remaining_heli_distance + 1e-9) {
                    weights[vidx] = 0.0;
                    continue;
                }

                int need_food = food_remaining[vidx];
                int need_other = other_remaining[vidx];

                // Calculate ratios (need vs availability in this trip)
                double ratio_p = (need_food > 0 && rem_p > 0) ? (double)need_food / rem_p : 0.0;
                double ratio_d = (need_food > 0 && rem_d > 0) ? (double)need_food / rem_d : 0.0;
                double ratio_o = (need_other > 0 && rem_o > 0) ? (double)need_other / rem_o : 0.0;

                // Build priority order
                vector<pair<double,char>> priorities = {
                    {ratio_p, 'p'}, {ratio_d, 'd'}, {ratio_o, 'o'}
                };
                sort(priorities.begin(), priorities.end(), [](auto &a, auto &b){
                    return a.first > b.first;
                });

                // Assign drops in priority order
                int give_p = 0, give_d = 0, give_o = 0;
                for (auto &pr : priorities) {
                    if (pr.second == 'p') {
                        give_p = min(rem_p, food_remaining[vidx]);
                    } else if (pr.second == 'd') {
                        give_d = min(rem_d, food_remaining[vidx] - give_p);
                    } else if (pr.second == 'o') {
                        give_o = min(rem_o, other_remaining[vidx]);
                    }
                }

                if (give_p + give_d + give_o == 0) {
                    weights[vidx] = 0.0;
                    continue;
                }

                Drop dr;
                dr.village_id = problem.villages[vidx].id;
                dr.perishable_food = give_p;
                dr.dry_food = give_d;
                dr.other_supplies = give_o;

                drops_for_trip.push_back(dr);

                rem_p -= give_p;
                rem_d -= give_d;
                rem_o -= give_o;
                food_remaining[vidx] -= (give_p + give_d);
                other_remaining[vidx] -= give_o;

                trip_dist = new_trip_dist;
                cur = vpt;

                if (food_remaining[vidx] <= 0 && other_remaining[vidx] <= 0) weights[vidx] = 0.0;
            }

            if (drops_for_trip.empty()) break;

            trip.drops = move(drops_for_trip);
            remaining_heli_distance -= trip_dist;
            plan.trips.push_back(move(trip));
        }
        sol[h.id-1] = move(plan);
    }

    CheckResult var = checkSolution(problem, sol);
    if (var.total_value < 0) {
        Solution mol;
        for (const auto &h : problem.helicopters) {
            HelicopterPlan plan;
            plan.helicopter_id = h.id;
            mol.push_back(plan);
        }
        return mol;
    }

    return sol;
}

// ----------------------- Local Search --------------------------------

Solution improveSolutionGreedy(
    const ProblemData &problem,Solution init,
    chrono::time_point<chrono::high_resolution_clock> start,
    chrono::seconds time_limit,
    int max_depth = 100, int neighbors_per_solution = 25,int beam_width = 10) {
    using clock = chrono::high_resolution_clock;
    mt19937 rng(random_device{}());

    struct ScoredSolution { Solution sol; double score; };

    // initial solution
    CheckResult cr_init = checkSolution(problem, init);
    double init_score = cr_init.valid ? cr_init.total_value : -1e18;
    vector<ScoredSolution> beam = { {init, init_score} };
    cout << "Starting beam greedy local improvement: initial score = " << init_score << "\n";
    double best_score = init_score;
    Solution best_sol = init;

    for (int depth = 0; depth < max_depth; ++depth) {
        if (clock::now() - start >= time_limit) {
            cout << "Time limit reached inside improveSolutionGreedy, returning best so far.\n";
            return best_sol;
        }

        vector<ScoredSolution> candidates;

        for (auto &entry : beam) {
            Solution &base = entry.sol;

            for (int k = 0; k < neighbors_per_solution; ++k) {
                if (clock::now() - start >= time_limit) {
                    cout << "Time limit reached during neighbor generation.\n";
                    return best_sol;
                }

                Solution cand = base;
                bool changed = false;

                typedef bool (*MoveFn)(const ProblemData&, Solution&,mt19937&);
                static vector<MoveFn> moves = {
                    move_reassign_within_trip,
                    move_swap_drops_between_trips,
                    move_relocate_trip_between_helicopters,
                    move_optimize_trip_route,
                    move_relocate_drop_position,
                    move_relocate_drop_to_other_trip_same_heli,
                    move_rebalance_pickups_same_heli,
                    move_swap_pickups_between_helicopters,
                    move_shuffle_trip_pickups,
                };

                int r = rndInt(0, (int)moves.size()-1, rng);
                changed = moves[r](problem, cand, rng);

                if (!changed) continue;

                CheckResult cr = checkSolution(problem, cand);
                if (!cr.valid) continue;

                candidates.push_back({cand, cr.total_value});

                if (cr.total_value > best_score) {
                    best_score = cr.total_value;
                    best_sol = cand;
                }
            }
        }

        if (candidates.empty()) break;

        sort(candidates.begin(), candidates.end(),
            [](const ScoredSolution &a, const ScoredSolution &b) {
                return a.score > b.score;
            });
        if ((int)candidates.size() > beam_width) candidates.resize(beam_width);

        beam = move(candidates);
    }
    cout << "Beam greedy improvement finished. best score = " << best_score << "\n";
    return best_sol;
}


// ----------------------- Main solve loop (unchanged but uses new functions) -----------------------

Solution solve(const ProblemData &problem) {
    random_device rd;
    mt19937 rng(rd());
    Solution best_solution;
    double best_score = -1e18;
    using clock = chrono::high_resolution_clock;
    auto start = clock::now();
    int max_time = problem.time_limit_minutes*60*0.95;
    auto time_limit = chrono::seconds(max_time);

    int it = 0;
    while (clock::now() - start < time_limit) {
        it++;

        // Step 1: generate random solution
        Solution candidate = randomSolutionGreedyValid(problem, start, time_limit);
        CheckResult cr = checkSolution(problem, candidate);
        if (!cr.valid) continue;  // skip invalid

        double score = cr.total_value;
        if (score < -1e17) continue;
        if (score > best_score) {
            // try greedy improvement only now
            candidate = improveSolutionGreedy(problem, candidate, start, time_limit);
            cr = checkSolution(problem, candidate);
            if (!cr.valid) continue;

            score = cr.total_value;

            // Step 3: update global best if improved
            if (score > best_score) {
                best_score = score;
                best_solution = candidate;
            }
        }
    }
    cout << "Final Best Score = " << best_score << "\n";
    return best_solution;
}
