#ifndef EXALT_HXX
#define EXALT_HXX

#include <fstream>
using std::ofstream;

#include <map>
using std::map;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;

#include <ios>
using std::hex;
using std::ios;

//for mkdir
#include <sys/stat.h>
#include <algorithm>

#include <chrono>

#include "rnn_genome.hxx"
#include "edge_pheromone.hxx"
#include "node_pheromone.hxx"
#include "rnn_genome.hxx"
#include "generate_nn.hxx"

#define PHEROMONES_THERESHOLD 20
#define PERIODIC_PHEROMONE_REDUCTION_RATIO 0.995
#define EXPLORATION_FACTOR 0.7          //Used in pick_node_type() which is I'm not using ( using the old_pick_node_type() )
#define INITIAL_PHEROMONE 0.7
#define MINIMUM_PHEROMONE_VALUE 0.2

class ACNNTO {
    private:
        int32_t ants;
        int32_t hidden_layers_depth;
        int32_t hidden_layer_nodes;
        double  pheromone_decay_parameter;
        double  pheromone_update_strength;
        double  pheromone_heuristic;

        double weight_reg_parameter;

        int32_t population_size;
        int32_t number_islands;
        vector<RNN_Genome*> population;

        vector<double> best_weights;

        map <int32_t, NODE_Pheromones*> colony;
        map <int32_t, EDGE_Pheromone*>  colony_edges;

        int32_t max_genomes;
        int32_t generated_genomes;
        int32_t inserted_genomes;
        int32_t total_bp_epochs;

        int32_t edge_innovation_count;
        int32_t node_innovation_count;

        map<string, int32_t> inserted_from_map;
        map<string, int32_t> generated_from_map;

        int32_t number_inputs;
        int32_t number_outputs;
        int32_t bp_iterations;
        double learning_rate;

        bool use_high_threshold;
        double high_threshold;

        bool use_low_threshold;
        double low_threshold;

        bool use_dropout;
        double dropout_probability;

        minstd_rand0 generator;
        uniform_real_distribution<double> rng_0_1;
        uniform_real_distribution<double> rng_crossover_weight;

        int32_t max_recurrent_depth;

        bool epigenetic_weights;
        double mutation_rate;
        double crossover_rate;
        double island_crossover_rate;

        double more_fit_crossover_rate;
        double less_fit_crossover_rate;

        double clone_rate;

        double add_edge_rate;
        double add_recurrent_edge_rate;
        double enable_edge_rate;
        double disable_edge_rate;
        double split_edge_rate;

        double add_node_rate;
        double enable_node_rate;
        double disable_node_rate;
        double split_node_rate;
        double merge_node_rate;

        vector<int> possible_node_types;

        string output_directory;
        ofstream *log_file;

        vector<string> input_parameter_names;
        vector<string> output_parameter_names;

        map<string,double> normalize_mins;
        map<string,double> normalize_maxs;
        map<int32_t, int32_t> node_types;


        ostringstream memory_log;

        string reward_type;
        string norm;
        bool bias_forward_paths;
        bool bias_edges;
        bool use_edge_inherited_weights;
        bool use_pheromone_weight_update;
        bool use_fitness_weight_update;
        bool use_two_ants_types;

        vector <double> edges_inherited_weights;
        vector <double> recedges_inherited_weights;

        double max_pheromone = PHEROMONES_THERESHOLD ;

        double phi = NULL ;
        double const_phi ;
        bool use_all_jumps ;
        bool use_forward_social_ants ;
        bool use_backward_social_ants ;

        std::chrono::time_point<std::chrono::system_clock> startClock;

        RNN_Node* check_node_existance(vector<RNN_Node_Interface*> &rnn_nodes,   int32_t node_inno);
        void check_edge_existance(vector<RNN_Edge*> &rnn_edges, int32_t innovation_number, RNN_Node* in_node, RNN_Node* out_node, double edge_weight);
        void check_recurrent_edge_existance(vector<RNN_Recurrent_Edge*> &recurrent_edges, int32_t innovation_number, int32_t depth, RNN_Node* in_node, RNN_Node* out_node, double edge_weight);
        // void check_recurrent_edge_existance_twoAnts(vector<RNN_Recurrent_Edge*> &recurrent_edges, int32_t innovation_number, int32_t depth, RNN_Node* in_node, RNN_Node* out_node, double edge_weight, int pre_recurrency_depth ) ;

        EDGE_Pheromone* pick_line(vector<EDGE_Pheromone*>* edges_pheromones);
        int old_pick_node_type(double* type_pheromones);
        int pick_node_type(double* type_pheromones);
        void reward_colony(RNN_Genome* g, bool reward);
        void reward_colony_regularization(RNN_Genome* g, bool reward);
        void reward_colony_fixed_rate(RNN_Genome* g, double treat_pheromone);
        void reward_colony_constant_rate(RNN_Genome* g, int treat_pheromone);
        void reduce_pheromones();
        void evaporate_pheromones();
        void new_evaporate_pheromones();
        double get_norm(RNN_Genome* g);

        void initialize_genome_parameters(RNN_Genome* genome );
        void build_social_colony(map <int32_t, NODE_Pheromones*>* social_colony, vector<RNN_Node_Interface*> rnn_nodes ) ;
        void build_social_colony(map <int32_t, NODE_Pheromones*>* social_colony, map <int32_t, EDGE_Pheromone*>* social_colony_edges, vector<RNN_Node_Interface*> rnn_nodes ) ;
        void build_layer_node_map(map<int32_t, vector<int32_t> >* layer_node_map) ;


    public:
        ACNNTO(int32_t _population_size, int32_t _max_genomes, const vector<string> &_input_parameter_names,
          const vector<string> &_output_parameter_names, const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs,
          int32_t _bp_iterations, double _learning_rate, bool _use_high_threshold, double _high_threshold, bool _use_low_threshold,
          double _low_threshold, string _output_directory, int32_t _ants, int32_t _hidden_layers_depth,
          int32_t _hidden_layer_nodes, double _pheromone_decay_parameter, double _pheromone_update_strength,
          double _pheromone_heuristic, int32_t _max_recurrent_depth, double _weight_reg_parameter);


        ACNNTO(int32_t _population_size, int32_t _max_genomes, const vector<string> &_input_parameter_names,
          const vector<string> &_output_parameter_names, const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs,
          int32_t _bp_iterations, double _learning_rate, bool _use_high_threshold, double _high_threshold, bool _use_low_threshold,
          double _low_threshold, string _output_directory, int32_t _ants, int32_t _hidden_layers_depth,
          int32_t _hidden_layer_nodes, double _pheromone_decay_parameter, double _pheromone_update_strength,
          double _pheromone_heuristic, int32_t _max_recurrent_depth, double _weight_reg_parameter, string _reward_type,
          string _norm, bool _bias_forward_paths, bool _bias_edges, bool _use_edge_inherited_weights,
          bool _use_pheromone_weight_update, bool _use_fitness_weight_update, bool _use_two_ants_types, double _const_phi, bool _use_all_jumps
          , bool _use_forward_social_ants, bool _use_backward_social_ants );

        ~ACNNTO();

        void print_population();
        void print_last_population();
        void write_memory_log(string filename);

        void set_possible_node_types(vector<string> possible_node_type_strings);

        int32_t population_contains(RNN_Genome* genome);
        bool populations_full() const;

        bool insert_genome(RNN_Genome* genome);

        int get_random_node_type();

        RNN_Genome* get_best_genome();
        RNN_Genome* get_worst_genome();

        double get_best_fitness();
        double get_worst_fitness();


        string get_output_directory() const;

        RNN_Genome* ants_march();
        RNN_Genome* explorer_social_ants_march();
        RNN_Genome* old_explorer_social_ants_march();

        void write_to_file(string bin_filename, bool verbose = false);
        void write_to_stream(ostream &bin_stream, bool verbose = false);

        // double get_genome_squared_weights_sum(RNN_Genome* genome);
        int32_t get_colony_number_edges();
        void bias_for_forward_paths();
        void bias_for_forward_rec_edges();
        void bias_for_edges();

        void set_colony_best_weights( RNN_Genome* g ) ;
        void pheromones_update_weights( RNN_Genome* genome ) ;
};

#endif
