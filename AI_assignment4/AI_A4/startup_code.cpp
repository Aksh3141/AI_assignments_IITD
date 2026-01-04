#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <map>
#include <iomanip>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <deque>
#include <set>
using namespace std;

class Graph_Node {
private:
    string Node_Name;
    vector<int> Children;
    vector<string> Parents;
    int nvalues;
    vector<string> values;
    vector<float> CPT;

public:
    Graph_Node(string name, int n, vector<string> vals) {
        Node_Name = name;
        nvalues = n;
        values = vals;
    }

    string get_name() const { return Node_Name; }
    vector<string> get_Parents() const { return Parents; }
    vector<float> get_CPT() const { return CPT; }
    int get_nvalues() const { return nvalues; }
    vector<string> get_values() const { return values; }

    void set_CPT(vector<float> new_CPT) {
        CPT.clear();
        CPT = new_CPT;
    }

    void set_Parents(vector<string> Parent_Nodes) {
        Parents.clear();
        Parents = Parent_Nodes;
    }

    int add_child(int new_child_index) {
        return 1;
    }
};

class network {
    list<Graph_Node> Pres_Graph;

public:
    int addNode(Graph_Node node) {
        Pres_Graph.push_back(node);
        return 0;
    }

    list<Graph_Node>::iterator get_nth_node(int n) {
        list<Graph_Node>::iterator listIt;
        int count = 0;
        for(listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++) {
            if(count == n) return listIt;
            count++;
        }
        return listIt;
    }

    list<Graph_Node>::iterator search_node(string val_name) {
        list<Graph_Node>::iterator listIt;
        for(listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++) {
            if(listIt->get_name().compare(val_name) == 0) return listIt;
        }
        return listIt;
    }

    int netSize() const { return Pres_Graph.size(); }

    int get_index(string val_name) {
        list<Graph_Node>::iterator listIt;
        int count = 0;
        for(listIt = Pres_Graph.begin(); listIt != Pres_Graph.end(); listIt++) {
            if(listIt->get_name().compare(val_name) == 0) return count;
            count++;
        }
        return -1;
    }

    list<Graph_Node>& get_graph() { return Pres_Graph; } 
};

string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

network read_network(const char* filename) {
    network BayesNet;
    string line;
    ifstream myfile(filename);
    
    if (!myfile.is_open()) {
        cout << "Error: Could not open file " << filename << endl;
        return BayesNet;
    }

    while (getline(myfile, line)) {
        line = trim(line);
        
        if (line.empty() || line[0] == '#') {
            continue;
        }

        stringstream ss(line);
        string token;
        ss >> token;

        if (token == "variable") {
            string var_name;
            ss >> var_name;
            
            getline(myfile, line);
            stringstream ss2(line);
            
            string type_keyword, discrete_keyword, bracket, equals;
            int num_values;
            ss2 >> type_keyword >> discrete_keyword >> bracket >> num_values >> bracket >> equals >> bracket;
            
            vector<string> values;
            string value;
            while (ss2 >> value) {
                if (value == "};") break;
                if (value.back() == ',') {
                    value = value.substr(0, value.length() - 1);
                }
                values.push_back(value);
            }
            
            Graph_Node new_node(var_name, num_values, values);
            BayesNet.addNode(new_node);
        }
        else if (token == "probability") {
            string paren, node_name;
            ss >> paren >> node_name;
            
            string full_prob_line = line;
            while (full_prob_line.find('{') == string::npos) {
                string next_line;
                if (!getline(myfile, next_line)) break;
                full_prob_line += " " + trim(next_line);
            }
            
            size_t start_paren = full_prob_line.find('(');
            size_t end_paren = full_prob_line.find(')');
            size_t pipe_pos = full_prob_line.find('|');
            
            if (start_paren == string::npos || end_paren == string::npos) {
                continue;
            }
            
            string prob_content = full_prob_line.substr(start_paren + 1, end_paren - start_paren - 1);
            stringstream prob_ss(prob_content);
            
            prob_ss >> node_name;
            
            list<Graph_Node>::iterator listIt = BayesNet.search_node(node_name);
            int index = BayesNet.get_index(node_name);
            
            vector<string> parents;
            
            if (pipe_pos != string::npos && pipe_pos < end_paren) {
                string parents_str = full_prob_line.substr(pipe_pos + 1, end_paren - pipe_pos - 1);
                stringstream parent_ss(parents_str);
                string parent;
                
                while (parent_ss >> parent) {
                    if (parent.back() == ',') {
                        parent = parent.substr(0, parent.length() - 1);
                    }
                    parents.push_back(parent);
                    
                    list<Graph_Node>::iterator parentIt = BayesNet.search_node(parent);
                    parentIt->add_child(index);
                }
            }
            
            listIt->set_Parents(parents);
            
            vector<float> cpt;
            bool reading_cpt = false;
            
            while (getline(myfile, line)) {
                line = trim(line);
                if (line == "};") break;
                if (line.empty()) continue;
                
                size_t close_paren = line.find(')');
                string prob_part;
                
                if (close_paren != string::npos) {
                    prob_part = line.substr(close_paren + 1);
                } else if (line.find("table") != string::npos) {
                    size_t table_pos = line.find("table");
                    prob_part = line.substr(table_pos + 5);
                } else {
                    prob_part = line;
                }
                
                stringstream ss_prob(prob_part);
                string token;
                while (ss_prob >> token) {
                    while (!token.empty() && (token.back() == ',' || token.back() == ';')) {
                        token = token.substr(0, token.length() - 1);
                    }
                    
                    if (!token.empty() && (isdigit(token[0]) || token[0] == '.' || token[0] == '-')) {
                        cpt.push_back(atof(token.c_str()));
                    }
                }
            }
            
            listIt->set_CPT(cpt);
        }
    }
    
    myfile.close();
    return BayesNet;
}

void write_network(const char* filename, network& BayesNet) {
    ofstream outfile(filename);

    if (!outfile.is_open()) {
        cout << "Error: Could not open file " << filename << " for writing" << endl;
        return;
    }

    outfile << "// Bayesian Network" << endl << endl;

    int N = BayesNet.netSize();

    for (int i = 0; i < N; i++) {
        auto node = BayesNet.get_nth_node(i);

        outfile << "variable " << node->get_name() << " {" << endl;
        outfile << "  type discrete [ " << node->get_nvalues() << " ] = { ";

        vector<string> vals = node->get_values();
        for (int j = 0; j < (int)vals.size(); j++) {
            outfile << vals[j];
            if (j < (int)vals.size() - 1) outfile << ", ";
        }
        outfile << " };" << endl;
        outfile << "}" << endl;
    }

    outfile << std::fixed << std::setprecision(6);
    for (int i = 0; i < N; i++) {
        auto node = BayesNet.get_nth_node(i);
        vector<string> parents = node->get_Parents();
        vector<string> values = node->get_values();
        vector<float> cpt = node->get_CPT();

        outfile << "probability ( " << node->get_name();
        if (!parents.empty()) {
            outfile << " | ";
            for (int j = 0; j < (int)parents.size(); j++) {
                outfile << parents[j];
                if (j < (int)parents.size() - 1) outfile << ", ";
            }
        }
        outfile << " ) {" << endl;

        vector<int> radices;
        radices.reserve(parents.size());
        for (auto &pname : parents) {
            auto pnode = BayesNet.search_node(pname);
            radices.push_back(pnode->get_nvalues());
        }

        int parent_combinations = 1;
        for (int r : radices) parent_combinations *= r;

        int cpt_index = 0;

        if (parents.empty()) {
            outfile << "    table ";
            for (int k = 0; k < (int)values.size(); k++) {
                if (cpt_index < (int)cpt.size()) outfile << cpt[cpt_index++];
                else outfile << "-1";
                if (k < (int)values.size() - 1) outfile << ", ";
            }
            outfile << ";" << endl;
        } else {
            for (int comb = 0; comb < parent_combinations; comb++) {
                vector<int> idx(parents.size(), 0);
                int tmp = comb;
                for (int p = (int)parents.size() - 1; p >= 0; p--) {
                    idx[p] = tmp % radices[p];
                    tmp /= radices[p];
                }

                outfile << "    ( ";
                for (int p = 0; p < (int)parents.size(); p++) {
                    auto pnode = BayesNet.search_node(parents[p]);
                    auto pvals = pnode->get_values();
                    int vidx = idx[p];
                    outfile << pvals[vidx];
                    if (p < (int)parents.size() - 1) outfile << ", ";
                }
                outfile << " ) ";

                for (int k = 0; k < (int)values.size(); k++) {
                    if (cpt_index < (int)cpt.size()) outfile << cpt[cpt_index++];
                    else outfile << "-1";
                    if (k < (int)values.size() - 1) outfile << ", ";
                }
                outfile << ";" << endl;
            }
        }

        outfile << "};" << endl << endl;
    }

    outfile.close();
    cout << "Network written to file: " << filename << endl;
}

typedef vector<string> DataRow;
typedef vector<DataRow> DataSet;

DataSet read_data(const char* filename) {
    DataSet data;
    ifstream myfile(filename);
    string line;

    if (!myfile.is_open()) {
        cerr << "Error: Could not open data file " << filename << endl;
        return data;
    }

    while (getline(myfile, line)) {
        stringstream ss(line);
        string cell;
        DataRow row;
        while (getline(ss, cell, ',')) {
            size_t first = cell.find_first_not_of(" \t\r\n\"");
            if (string::npos == first) continue;
            size_t last = cell.find_last_not_of(" \t\r\n\"");
            row.push_back(cell.substr(first, (last - first + 1)));
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    myfile.close();
    return data;
}

float infer_probability(network& bn, const string& query_var, const string& query_val, 
                        const map<string, string>& evidence) {
    auto node = bn.search_node(query_var);
    if (node == bn.get_graph().end()) return 0.0f;
    return 1.0f / node->get_nvalues(); 
}

void initialize_unknown_cpts(network& BayesNet) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> distrib(0.01, 0.99);

    for (auto node_it = BayesNet.get_graph().begin(); node_it != BayesNet.get_graph().end(); ++node_it) {
        vector<float> cpt = node_it->get_CPT();
        vector<string> parents = node_it->get_Parents();
        int n_vals = node_it->get_nvalues();

        vector<int> radices;
        for (const string& pname : parents) {
            radices.push_back(BayesNet.search_node(pname)->get_nvalues());
        }
        int parent_combinations = accumulate(radices.begin(), radices.end(), 1, multiplies<int>());
        int total_entries = parent_combinations * n_vals;

        if (cpt.size() != total_entries) cpt.resize(total_entries, -1.0f); 
        
        for (int i = 0; i < parent_combinations; ++i) {
            float sum = 0.0f;
            int unknown_count = 0;
            vector<int> unknown_indices;

            for (int j = 0; j < n_vals; ++j) {
                int cpt_idx = i * n_vals + j;
                if (cpt[cpt_idx] == -1.0f) {
                    unknown_count++;
                    unknown_indices.push_back(cpt_idx);
                } else {
                    sum += cpt[cpt_idx];
                }
            }

            if (unknown_count > 0) {
                float remaining_prob = 1.0f - sum;
                if (remaining_prob < 0) remaining_prob = 0.0f;

                if (unknown_count == n_vals) {
                    float uniform_prob = 1.0f / n_vals;
                    for (int idx : unknown_indices) {
                        cpt[idx] = uniform_prob; 
                    }
                } else {
                    float current_sum = 0.0f;
                    for (size_t k = 0; k < unknown_indices.size(); ++k) {
                        float p = (k == unknown_indices.size() - 1) 
                                  ? (remaining_prob - current_sum) 
                                  : remaining_prob * (distrib(gen) * (1.0f / (n_vals * 2)));

                        p = max(0.0f, p);
                        cpt[unknown_indices[k]] = p;
                        current_sum += p;
                    }
                    float final_sum = 0.0f;
                    for(int j = 0; j < n_vals; ++j) final_sum += cpt[i * n_vals + j];
                    if (final_sum > 0.0f) {
                        float factor = 1.0f / final_sum;
                        for(int j = 0; j < n_vals; ++j) cpt[i * n_vals + j] *= factor;
                    }
                }
            }
        }
        node_it->set_CPT(cpt);
    }
}

int get_cpt_index(network& BayesNet, const Graph_Node* node, const string& value_name, const map<string, string>& parent_config) {
    const vector<string>& parents = node->get_Parents();
    int n_vals = node->get_nvalues();

    const vector<string>& node_vals = node->get_values();
    int value_idx = -1;
    for (int i = 0; i < (int)node_vals.size(); ++i) {
        if (node_vals[i] == value_name) {
            value_idx = i;
            break;
        }
    }
    if (value_idx == -1) return -1; 

    if (parents.empty()) {
        return value_idx;
    }

    int index_offset = 0;
    int multiplier = 1;
    
    for (int p = (int)parents.size() - 1; p >= 0; p--) {
        const string& pname = parents[p];
        
        if (parent_config.find(pname) == parent_config.end()) {
            return -2; 
        }
        const string& pval = parent_config.at(pname);
        
        auto pnode = BayesNet.search_node(pname);
        const vector<string>& pvals = pnode->get_values();
        
        int pval_idx = -1;
        for (int i = 0; i < (int)pvals.size(); ++i) {
            if (pvals[i] == pval) {
                pval_idx = i;
                break;
            }
        }
        
        if (pval_idx == -1) return -3;

        index_offset += pval_idx * multiplier;
        multiplier *= pnode->get_nvalues();
    }
    
    return (index_offset * n_vals) + value_idx;
}

#include <chrono> 

void EM_learn_parameters(network& BayesNet, DataSet& data, int max_iter = 500, float tolerance = 1e-12) {
    const double MIN_PROB = 1e-15;
    const double LAPLACE_ALPHA = 0.5;
    const int MAX_DURATION_SEC = 110;
    const int CONVERGENCE_WINDOW = 5; 
    const double ADAPTIVE_LEARNING_RATE = 0.99;  
    
    int N = BayesNet.netSize();

    // --- Cache node info ---
    vector<Graph_Node*> nodes(N);
    vector<string> node_names(N);
    vector<vector<string>> node_values(N);
    vector<vector<string>> node_parents(N);
    vector<vector<int>> parent_indices(N);
    vector<int> nvalues(N);
    vector<int> cpt_sizes(N);
    vector<int> parent_combinations(N);
    vector<vector<int>> parent_radices(N);
    vector<map<string, int>> value_to_index(N);
    
    int max_domain_size = 0;
    int total_cpt_entries = 0;
    
    for (int i = 0; i < N; ++i) {
        nodes[i] = &(*BayesNet.get_nth_node(i));
        node_names[i] = nodes[i]->get_name();
        node_values[i] = nodes[i]->get_values();
        node_parents[i] = nodes[i]->get_Parents();
        nvalues[i] = nodes[i]->get_nvalues();
        max_domain_size = max(max_domain_size, nvalues[i]);

        for (int v = 0; v < nvalues[i]; ++v) {
            value_to_index[i][node_values[i][v]] = v;}

        parent_indices[i].reserve(node_parents[i].size());
        parent_radices[i].reserve(node_parents[i].size());
        for (const string& pname : node_parents[i]) {
            int pidx = BayesNet.get_index(pname);parent_indices[i].push_back(pidx);parent_radices[i].push_back(nvalues[pidx]);}
        int combos = 1;
        for (int r : parent_radices[i]) combos *= r;parent_combinations[i] = combos;cpt_sizes[i] = combos * nvalues[i];
        total_cpt_entries += cpt_sizes[i];}

    // --- Precompute children and Markov blanket ---
    vector<vector<int>> children_of(N);
    vector<vector<int>> markov_blanket(N);
    
    for (int child = 0; child < N; ++child) {
        for (int p : parent_indices[child]) {
            children_of[p].push_back(child);}}
    
    // Build Markov blanket for each node (parents + children + children's parents)
    for (int i = 0; i < N; ++i) {
        set<int> mb_set;
        // Add parents
        for (int p : parent_indices[i]) mb_set.insert(p);
        // Add children
        for (int ch : children_of[i]) {
            mb_set.insert(ch);
            // Add children's parents
            for (int cp : parent_indices[ch]) mb_set.insert(cp);
        }
        mb_set.erase(i);
        markov_blanket[i].assign(mb_set.begin(), mb_set.end());
    }

    // --- Convert data to integer indices with full validation ---
    vector<vector<int>> int_data;
    int_data.reserve(data.size());
    
    int total_missing = 0;
    int total_records = 0;
    map<int, int> missing_per_node;
    
    for (size_t r = 0; r < data.size(); ++r) {
        vector<int> rec(N);
        bool valid = true;
        int missing_count = 0;
        
        for (int i = 0; i < N; ++i) {
            const string& val = data[r][i];
            if (val == "?" || val == "\"?\"") {
                rec[i] = -1;
                missing_count++;
                missing_per_node[i]++;
            } else {
                auto it = value_to_index[i].find(val);
                if (it != value_to_index[i].end()) {
                    rec[i] = it->second;
                } else {
                    valid = false;
                    break;
                }
            }
        }
        
        // Only use records with 0 or 1 missing values for quality
        if (valid && missing_count <= 1) {
            int_data.push_back(rec);
            total_missing += missing_count;
            total_records++;
        }
    }
    
    cout << "Using " << total_records << "/" << data.size() << " records, " 
         << total_missing << " missing values" << endl;

    // --- Fast CPT indexing ---
    auto compute_cpt_index = [&](int node_idx, int value_idx, const vector<int>& parent_vals) -> int {
        int combo_idx = 0;
        int multiplier = 1;
        const auto& pradices = parent_radices[node_idx];
        for (int p = (int)pradices.size() - 1; p >= 0; --p) {
            combo_idx += parent_vals[p] * multiplier;
            multiplier *= pradices[p];
        }
        return combo_idx * nvalues[node_idx] + value_idx;
    };

    // === SMART INITIALIZATION: Count-based + Smoothing ===
    cout << "Initializing CPTs from data statistics..." << endl;
    
    vector<vector<double>> empirical_counts(N);
    for (int i = 0; i < N; ++i) {
        empirical_counts[i].assign(cpt_sizes[i], LAPLACE_ALPHA);
    }
    
    // First pass: collect counts from fully observed records
    for (const auto& record : int_data) {
        bool fully_observed = true;
        for (int val : record) {
            if (val == -1) {
                fully_observed = false;
                break;
            }
        }
        
        if (fully_observed) {
            for (int ni = 0; ni < N; ++ni) {
                vector<int> pvals;
                for (int p : parent_indices[ni]) {
                    pvals.push_back(record[p]);
                }
                int idx = compute_cpt_index(ni, record[ni], pvals);
                empirical_counts[ni][idx] += 1.0;
            }
        }
    }
    
    // Initialize CPTs from empirical counts
    for (int i = 0; i < N; ++i) {
        int n_vals = nvalues[i];
        int combos = parent_combinations[i];
        vector<float> cpt(cpt_sizes[i]);
        
        for (int pc = 0; pc < combos; ++pc) {
            double sum = 0.0;
            int base = pc * n_vals;
            for (int k = 0; k < n_vals; ++k) {
                sum += empirical_counts[i][base + k];
            }
            if (sum < MIN_PROB) sum = n_vals * LAPLACE_ALPHA;
            
            for (int k = 0; k < n_vals; ++k) {
                cpt[base + k] = empirical_counts[i][base + k] / sum;
            }
        }
        nodes[i]->set_CPT(cpt);
    }

    using namespace std::chrono;
    auto start_time = steady_clock::now();
    const auto MAX_DURATION = seconds(MAX_DURATION_SEC);

    // Pre-allocate buffers
    vector<double> family_prob_buffer(max_domain_size);
    vector<int> parent_vals_buffer;
    parent_vals_buffer.reserve(15);
    
    // Track convergence history
    deque<double> change_history;
    vector<vector<double>> prev_cpts(N);
    vector<vector<double>> momentum(N);  // For smoother updates
    
    for (int i = 0; i < N; ++i) {
        const auto& cpt = nodes[i]->get_CPT();
        prev_cpts[i].assign(cpt.begin(), cpt.end());
        momentum[i].assign(cpt_sizes[i], 0.0);
    }

    int iter = 0;
    double best_change = 1e10;
    int no_improve_count = 0;
    
    while (iter < max_iter) {
        auto now = steady_clock::now();
        if (now - start_time > MAX_DURATION) {
            cout << "\nTime limit reached after " << iter << " iterations.\n";
            break;
        }

        iter++;

        // === E-STEP with HIGH-PRECISION ACCUMULATION ===
        vector<vector<double>> expected_counts(N);
        for (int i = 0; i < N; ++i) {
            expected_counts[i].assign(cpt_sizes[i], LAPLACE_ALPHA);
        }

        for (size_t rec = 0; rec < int_data.size(); ++rec) {
            const auto& record = int_data[rec];
            
            int missing_idx = -1;
            for (int i = 0; i < N; ++i) {
                if (record[i] == -1) { 
                    missing_idx = i; 
                    break; 
                }
            }

            if (missing_idx == -1) {
                // Fully observed
                for (int ni = 0; ni < N; ++ni) {
                    parent_vals_buffer.clear();
                    for (int p : parent_indices[ni]) {
                        parent_vals_buffer.push_back(record[p]);
                    }
                    int cpt_idx = compute_cpt_index(ni, record[ni], parent_vals_buffer);
                    expected_counts[ni][cpt_idx] += 1.0;
                }
                continue;
            }

            // === HANDLE MISSING VALUE ===
            const int miss = missing_idx;
            const int miss_domain_size = nvalues[miss];

            // Family = miss + children only (more efficient)
            vector<char> is_in_family(N, 0);
            is_in_family[miss] = 1;
            for (int ch : children_of[miss]) {
                is_in_family[ch] = 1;
            }

            // Compute log probabilities for numerical stability
            vector<double> log_family_probs(miss_domain_size, -1e100);
            double max_log_prob = -1e100;

            for (int mv = 0; mv < miss_domain_size; ++mv) {
                double log_prob = 0.0;
                bool valid = true;

                // P(miss = mv | parents)
                {
                    parent_vals_buffer.clear();
                    for (int p : parent_indices[miss]) {
                        parent_vals_buffer.push_back(record[p]);
                    }
                    int idx = compute_cpt_index(miss, mv, parent_vals_buffer);
                    double p = nodes[miss]->get_CPT()[idx];
                    if (p < MIN_PROB) {
                        valid = false;
                    } else {
                        log_prob += log(p);
                    }
                }

                // P(children | miss = mv, other parents)
                if (valid) {
                    for (int ch : children_of[miss]) {
                        parent_vals_buffer.clear();
                        for (int p : parent_indices[ch]) {
                            parent_vals_buffer.push_back((p == miss) ? mv : record[p]);
                        }
                        int idx = compute_cpt_index(ch, record[ch], parent_vals_buffer);
                        double p = nodes[ch]->get_CPT()[idx];
                        if (p < MIN_PROB) {
                            valid = false;
                            break;
                        }
                        log_prob += log(p);
                    }
                }

                if (valid) {
                    log_family_probs[mv] = log_prob;
                    if (log_prob > max_log_prob) {
                        max_log_prob = log_prob;
                    }
                }
            }

            if (max_log_prob == -1e100) continue;

            // Normalize in probability space with numerical stability
            double sum_probs = 0.0;
            for (int mv = 0; mv < miss_domain_size; ++mv) {
                if (log_family_probs[mv] > -1e50) {
                    family_prob_buffer[mv] = exp(log_family_probs[mv] - max_log_prob);
                    sum_probs += family_prob_buffer[mv];
                } else {
                    family_prob_buffer[mv] = 0.0;
                }
            }

            if (sum_probs < MIN_PROB) continue;

            // Accumulate expected counts
            for (int ni = 0; ni < N; ++ni) {
                if (!is_in_family[ni]) {
                    parent_vals_buffer.clear();
                    for (int p : parent_indices[ni]) {
                        parent_vals_buffer.push_back(record[p]);
                    }
                    int idx = compute_cpt_index(ni, record[ni], parent_vals_buffer);
                    expected_counts[ni][idx] += 1.0;
                } else {
                    for (int mv = 0; mv < miss_domain_size; ++mv) {
                        double weight = family_prob_buffer[mv] / sum_probs;
                        if (weight < MIN_PROB) continue;

                        parent_vals_buffer.clear();
                        for (int p : parent_indices[ni]) {
                            parent_vals_buffer.push_back((p == miss) ? mv : record[p]);
                        }
                        int val = (ni == miss) ? mv : record[ni];
                        int idx = compute_cpt_index(ni, val, parent_vals_buffer);
                        expected_counts[ni][idx] += weight;
                    }
                }
            }
        }

        // === M-STEP WITH MOMENTUM ===
        double max_change = 0.0;
        double total_change = 0.0;
        
        for (int i = 0; i < N; ++i) {
            int n_vals = nvalues[i];
            int combos = parent_combinations[i];
            vector<float> new_cpt(cpt_sizes[i]);

            for (int pc = 0; pc < combos; ++pc) {
                int base = pc * n_vals;
                double total = 0.0;
                for (int k = 0; k < n_vals; ++k) {total += expected_counts[i][base + k];}
                
                if (total < MIN_PROB) total = n_vals * LAPLACE_ALPHA;
                
                for (int k = 0; k < n_vals; ++k) {
                    double raw_update = expected_counts[i][base + k] / total;
                    
                    // Apply momentum for smoother convergence
                    double old_val = prev_cpts[i][base + k];
                    double delta = raw_update - old_val;
                    momentum[i][base + k] = ADAPTIVE_LEARNING_RATE * momentum[i][base + k] + 
                                            (1.0 - ADAPTIVE_LEARNING_RATE) * delta;
                    double new_val = old_val + momentum[i][base + k];
                    new_val = max(MIN_PROB, min(1.0 - MIN_PROB, new_val));
                    
                    new_cpt[base + k] = new_val;
                    
                    double change = fabs(new_val - old_val);
                    max_change = max(max_change, change);
                    total_change += change;
                }
                
                // Re-normalize after momentum
                double sum = 0.0;
                for (int k = 0; k < n_vals; ++k) {
                    sum += new_cpt[base + k];
                }
                if (sum > MIN_PROB) {
                    for (int k = 0; k < n_vals; ++k) {
                        new_cpt[base + k] /= sum;
                    }
                }
            }
            
            nodes[i]->set_CPT(new_cpt);
            
            for (int j = 0; j < cpt_sizes[i]; ++j) {
                prev_cpts[i][j] = new_cpt[j];
            }
        }

        double avg_change = total_change / total_cpt_entries;
        
        // === CONVERGENCE DETECTION ===
        change_history.push_back(avg_change);
        if (change_history.size() > CONVERGENCE_WINDOW) {
            change_history.pop_front();
        }
        
        if (iter % 10 == 0 || avg_change < tolerance * 10) {
            cout << "Iter " << iter << ": avg_change=" << avg_change 
                 << ", max_change=" << max_change << endl;
        }
        
        // Check for convergence
        if (avg_change < tolerance) {
            cout << "Converged at iteration " << iter << " (avg_change=" << avg_change << ")\n";
            break;
        }
        
        // Check for stability over window
        if (change_history.size() == CONVERGENCE_WINDOW) {
            double window_max = *max_element(change_history.begin(), change_history.end());
            double window_min = *min_element(change_history.begin(), change_history.end());
            if (window_max - window_min < tolerance * 2) {
                cout << "Stable convergence at iteration " << iter << "\n";
                break;
            }
        }
        
        //Early stopping with patience
        if (avg_change < best_change * 0.995) {
            best_change = avg_change;
            no_improve_count = 0;
        } else {
            no_improve_count++;
            if (no_improve_count >= 15) {
                cout << "Early stopping at iteration " << iter << " (no improvement)\n";
                break;
            }
        }
    }

    // === FINAL HIGH-PRECISION NORMALIZATION ===
    cout << "Final CPT normalization..." << endl;
    for (int i = 0; i < N; ++i) {
        vector<float> cpt = nodes[i]->get_CPT();
        int n_vals = nvalues[i];
        int combos = parent_combinations[i];
        
        for (int pc = 0; pc < combos; ++pc) {
            int start = pc * n_vals;
            
            // High-precision normalization
            double sum = 0.0;
            for (int k = 0; k < n_vals; ++k) {
                sum += cpt[start + k];
            }
            
            if (sum > MIN_PROB) {
                // Normalize
                for (int k = 0; k < n_vals; ++k) {
                    cpt[start + k] = cpt[start + k] / sum;
                }
                
                // Round to 4 decimals
                for (int k = 0; k < n_vals; ++k) {
                    cpt[start + k] = round(cpt[start + k] * 10000.0) / 10000.0;
                }
                
                // Final exact normalization
                sum = 0.0;
                for (int k = 0; k < n_vals; ++k) {
                    sum += cpt[start + k];
                }
                
                if (fabs(sum - 1.0) > 1e-6) {
                    // Distribute error to maintain sum = 1.0
                    double error = 1.0 - sum;
                    int max_idx = 0;
                    for (int k = 1; k < n_vals; ++k) {
                        if (cpt[start + k] > cpt[start + max_idx]) {
                            max_idx = k;
                        }
                    }
                    cpt[start + max_idx] += error;
                }
            }
        }
        nodes[i]->set_CPT(cpt);
    }
    
    cout << "EM complete: " << iter << " iterations" << endl;
}



#ifndef BN_LIB
int main(int argc, char* argv[]) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " <input_bif_file> <input_data_file> <output_bif_file>" << endl;
        return 1;
    }
    
    network BayesNet = read_network(argv[1]);
    DataSet data = read_data(argv[2]);
    
    if (BayesNet.netSize() == 0 || data.empty()) {
        cerr << "Error loading network or data." << endl;
        return 1;
    }

    cout << "Starting EM learning..." << endl;
    
    EM_learn_parameters(BayesNet, data);
    
    write_network(argv[3], BayesNet);

    return 0;
}
#endif // BN_LIB
