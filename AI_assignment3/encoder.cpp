#include <iostream>
#include <vector>
#include <string>
#include <numeric>

using namespace std;


// Represents the metro routing problem

struct MetroScenario {
    int scenario;
    int N, M, K, J, P;
    vector<pair<int,int>> start_positions;  // Each agent's start {row,col}
    vector<pair<int,int>> end_positions;    // Each agent's end {row,col}
    vector<pair<int,int>> popular_cells;    // Only used in scenario 2
};


// Maps logical metro/row/col to CNF variable IDs

struct CNFMapper {
    int N, M, K, J;
    int p_off, h_off, v_off, t_off, s_off;
    int total_vars;

    CNFMapper(const MetroScenario& prob) : N(prob.N), M(prob.M), K(prob.K), J(prob.J) {
        int cells = M * N;
        int p_cnt = K * cells;
        int h_cnt = K * M * (N-1);
        int v_cnt = K * (M-1) * N;
        int t_cnt = K * cells;
        int s_cnt = K * (cells + 1) * (J + 2);

        p_off = 1;
        h_off = p_off + p_cnt;
        v_off = h_off + h_cnt;
        t_off = v_off + v_cnt;
        s_off = t_off + t_cnt;

        total_vars = p_cnt + h_cnt + v_cnt + t_cnt + s_cnt;
    }

    int p(int k,int r,int c) const { return p_off + k*(M*N) + r*N + c; }
    int h(int k,int r,int c) const { return h_off + k*(M*(N-1)) + r*(N-1) + c; }
    int v(int k,int r,int c) const { return v_off + k*((M-1)*N) + r*N + c; }
    int t(int k,int r,int c) const { return t_off + k*(M*N) + r*N + c; }
    int s(int k,int idx,int j) const { return s_off + k*((M*N+1)*(J+2)) + idx*(J+2) + j; }
};


// Collects CNF clauses for later output

struct CNFCollector {
    vector<vector<int>> clauses;
    void push_clause(const vector<int>& c) { if(!c.empty()) clauses.push_back(c); }
};


// Add at-most-one constraint for a set of variables

void at_most_one(CNFCollector& cm, const vector<int>& vars) {
    for(size_t i=0;i<vars.size();++i)
        for(size_t j=i+1;j<vars.size();++j)
            cm.push_clause({-vars[i],-vars[j]});
}


// Main function

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    MetroScenario prob;
    cin >> prob.scenario;
    if(prob.scenario == 1) {
        cin >> prob.N >> prob.M >> prob.K >> prob.J;
        prob.P = 0;
    } else {
        cin >> prob.N >> prob.M >> prob.K >> prob.J >> prob.P;
    }

    // Read start and end positions
    prob.start_positions.resize(prob.K);
    prob.end_positions.resize(prob.K);
    for(int k=0;k<prob.K;++k) {
        cin >> prob.start_positions[k].second >> prob.start_positions[k].first;
        cin >> prob.end_positions[k].second >> prob.end_positions[k].first;
    }

    // Read popular cells if scenario 2
    if(prob.scenario == 2) {
        prob.popular_cells.resize(prob.P);
        for(int i=0;i<prob.P;++i)
            cin >> prob.popular_cells[i].second >> prob.popular_cells[i].first;
    }

    CNFMapper mapper(prob);
    CNFCollector cnf;

   
    // Constraint 1: at most one agent per cell
  
    for(int r=0;r<prob.M;++r) {
        for(int c=0;c<prob.N;++c) {
            vector<int> cell_vars;
            for(int k=0;k<prob.K;++k) cell_vars.push_back(mapper.p(k,r,c));
            at_most_one(cnf, cell_vars);
        }
    }

    
    // Agent path constraints
    
    for(int k=0;k<prob.K;++k) {
        int sr=prob.start_positions[k].first, sc=prob.start_positions[k].second;
        int er=prob.end_positions[k].first,   ec=prob.end_positions[k].second;

        // Start/end cells must be visited
        cnf.push_clause({mapper.p(k,sr,sc)});
        cnf.push_clause({mapper.p(k,er,ec)});

        // Flow and degree constraints for each cell
        for(int r=0;r<prob.M;++r) {
            for(int c=0;c<prob.N;++c) {
                int p_var = mapper.p(k,r,c);
                vector<int> adj_edges;
                if(c>0) adj_edges.push_back(mapper.h(k,r,c-1));
                if(c<prob.N-1) adj_edges.push_back(mapper.h(k,r,c));
                if(r>0) adj_edges.push_back(mapper.v(k,r-1,c));
                if(r<prob.M-1) adj_edges.push_back(mapper.v(k,r,c));

                // If p is true, at least one adjacent edge must exist
                vector<int> clause = {-p_var};
                for(int e:adj_edges) clause.push_back(e);
                cnf.push_clause(clause);

                // Active edge implies cell occupied
                for(int e:adj_edges) cnf.push_clause({-e,p_var});

                // Degree constraints
                if((r==sr && c==sc) || (r==er && c==ec)) {
                    // Start/end: exactly one edge
                    cnf.push_clause(adj_edges);
                    at_most_one(cnf, adj_edges);
                } else {
                    // Internal: degree ≤2
                    if(adj_edges.size()>=3)
                        for(size_t i=0;i<adj_edges.size();++i)
                            for(size_t j=i+1;j<adj_edges.size();++j)
                                for(size_t l=j+1;l<adj_edges.size();++l)
                                    cnf.push_clause({-p_var,-adj_edges[i],-adj_edges[j],-adj_edges[l]});
                    // p true => at least two edges
                    if(adj_edges.size()>=2)
                        for(size_t i=0;i<adj_edges.size();++i){
                            vector<int> temp = {-p_var};
                            for(size_t j=0;j<adj_edges.size();++j) if(i!=j) temp.push_back(adj_edges[j]);
                            cnf.push_clause(temp);
                        }
                }
            }
        }

        
        // Turn detection
        
        vector<int> turns;
        for(int r=0;r<prob.M;++r) for(int c=0;c<prob.N;++c) {
            int t = mapper.t(k,r,c);
            turns.push_back(t);

            int h_in  = (c>0) ? mapper.h(k,r,c-1) : 0;
            int h_out = (c<prob.N-1) ? mapper.h(k,r,c) : 0;
            int v_in  = (r>0) ? mapper.v(k,r-1,c) : 0;
            int v_out = (r<prob.M-1) ? mapper.v(k,r,c) : 0;

            cnf.push_clause({-t,mapper.p(k,r,c)});
            if(h_in && v_in) cnf.push_clause({-h_in,-v_in,t});
            if(h_in && v_out) cnf.push_clause({-h_in,-v_out,t});
            if(h_out && v_in) cnf.push_clause({-h_out,-v_in,t});
            if(h_out && v_out) cnf.push_clause({-h_out,-v_out,t});
        }


        // Sequential counter for turn limit

        int total_cells = prob.M*prob.N;
        for(int j=1;j<=prob.J+1;++j) cnf.push_clause({-mapper.s(k,0,j)});

        for(int i=1;i<=total_cells;++i){
            int t_i = turns[i-1];
            cnf.push_clause({-mapper.s(k,i-1,1), mapper.s(k,i,1)});
            cnf.push_clause({-t_i, mapper.s(k,i,1)});

            for(int j=2;j<=prob.J+1;++j){
                cnf.push_clause({-mapper.s(k,i-1,j), mapper.s(k,i,j)});
                cnf.push_clause({-t_i, -mapper.s(k,i-1,j-1), mapper.s(k,i,j)});
            }
        }
        cnf.push_clause({-mapper.s(k,total_cells,prob.J+1)});
    }

    // Popular cell constraints (Scenario 2)
    if(prob.scenario==2){
        for(auto& cell: prob.popular_cells){
            vector<int> clause;
            for(int k=0;k<prob.K;++k) clause.push_back(mapper.p(k,cell.first,cell.second));
            cnf.push_clause(clause);
            // Optional: at_most_one(cnf, clause);
        }
    }

    // Output DIMACS CNF
    cout << "p cnf " << mapper.total_vars << " " << cnf.clauses.size() << "\n";
    for(auto& clause: cnf.clauses){
        for(int v: clause) cout << v << " ";
        cout << "0\n";
    }

    return 0;
}
