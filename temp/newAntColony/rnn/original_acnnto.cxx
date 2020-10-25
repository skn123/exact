/*
-- This class is for ants to choose paths along the neural network structure.
-- The class will generate a fixed structure for the neural network and then make copies for
   the ants to march over and select their paths.
-- As the class generates the neural network, it will also generate a colony which will hold the
   pheromones of the ants.
-- Ants will also pick a the nodes types for the neural network as they march through the neural network.
-- Class can reward the fit genomes (colonies) and increase their phermones. It can also penalize them.
-- The class will also decrease the phermones periodically as throught the iterations.
**--AbdElRahman--**
*/


#include <algorithm>
using std::sort;

#include <chrono>

#include <stdlib.h>

#include <fstream>
using std::ofstream;

#include <iomanip>
using std::setw;
using std::setprecision;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include<map>
using std::map;

#include <string>
using std::string;
using std::to_string;

//for mkdir
#include <sys/stat.h>

#include <cmath>

#include "acnnto.hxx"
#include "edge_pheromone.hxx"
#include "node_pheromone.hxx"
#include "rnn_genome.hxx"
#include "generate_nn.hxx"

ACNNTO::~ACNNTO() {
    RNN_Genome *genome;
    while (population.size()>0){
      genome = population.back();
      delete genome;
      population.pop_back();
    }
    int c_size = colony.size();
    for (int i=-1; i<c_size; i++){
        if ( i == colony.size()-1){
            i*=-1;
        }
        cout<<"I: "<<i<<endl;
        int p_size = colony[i]->pheromone_lines->size();
        for ( int j=0; j<p_size; j++){
            delete colony[i]->pheromone_lines->at(j);
        }
        delete colony[i]->pheromone_lines;
        delete colony[i]->type_pheromones;
    }
    for (int i=-1; i<c_size; i++){
        if ( i == colony.size()-1){
            i*=-1;
        }
        colony.erase(i);
    }
    cout<<"INFO: CLEANED CLONY!"<<endl;
}



ACNNTO::ACNNTO(int32_t _population_size, int32_t _max_genomes, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names, const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs, int32_t _bp_iterations, double _learning_rate, bool _use_high_threshold, double _high_threshold, bool _use_low_threshold, double _low_threshold, string _output_directory, int32_t _ants, int32_t _hidden_layers_depth, int32_t _hidden_layer_nodes, double _pheromone_decay_parameter, double _pheromone_update_strength, double _pheromone_heuristic) : population_size(_population_size), max_genomes(_max_genomes), number_inputs(_input_parameter_names.size()), number_outputs(_output_parameter_names.size()), bp_iterations(_bp_iterations), learning_rate(_learning_rate), use_high_threshold(_use_high_threshold), high_threshold(_high_threshold), use_low_threshold(_use_low_threshold), low_threshold(_low_threshold), output_directory(_output_directory), ants(_ants), hidden_layers_depth(_hidden_layers_depth), hidden_layer_nodes(_hidden_layer_nodes), pheromone_decay_parameter(_pheromone_decay_parameter), pheromone_update_strength(_pheromone_update_strength), pheromone_heuristic(_pheromone_heuristic) {

    input_parameter_names   = _input_parameter_names;
    output_parameter_names  = _output_parameter_names;
    normalize_mins          = _normalize_mins;
    normalize_maxs          = _normalize_maxs;

    inserted_genomes  = 0;
    generated_genomes = 0;
    total_bp_epochs   = 0;

    edge_innovation_count = 0;
    node_innovation_count = 0;

    // population = vector<RNN_Genome*>;

    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

    rng_crossover_weight = uniform_real_distribution<double>(-0.5, 1.5);

    max_recurrent_depth = 3;

    epigenetic_weights = true;


    possible_node_types.clear();
    possible_node_types.push_back(FEED_FORWARD_NODE);
    possible_node_types.push_back(JORDAN_NODE);
    possible_node_types.push_back(ELMAN_NODE);
    possible_node_types.push_back(UGRNN_NODE);
    possible_node_types.push_back(MGU_NODE);
    possible_node_types.push_back(GRU_NODE);
    possible_node_types.push_back(LSTM_NODE);
    possible_node_types.push_back(DELTA_NODE);

    node_types[FEED_FORWARD_NODE]   = 0;
    node_types[UGRNN_NODE]          = 1;
    node_types[MGU_NODE]            = 2;
    node_types[GRU_NODE]            = 3;
    node_types[DELTA_NODE]          = 4;
    node_types[LSTM_NODE]           = 5;


    if (output_directory != "") {
        mkdir(output_directory.c_str(), 0777);
        log_file = new ofstream(output_directory + "/" + "fitness_log.csv");
        (*log_file) << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges" << endl;
        memory_log << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges" << endl;
    } else {
        log_file = NULL;
    }

    startClock = std::chrono::system_clock::now();

    // create_colony_pheromones(number_inputs, 0, 0, number_outputs, max_recurrent_depth, colony);
    // cout<<"aColony Size: "<<colony.size()<<endl;
    create_colony_pheromones(number_inputs, 0, 0, number_outputs, max_recurrent_depth, colony);
    // colony = extra(number_inputs, 1, 6, number_outputs, max_recurrent_depth, colony, test);

    // cout<<"bColony Size: "<<colony.size()<<endl;
    // cout<<"ANT COLONY:: Index of first element in map"<<colony[-1]<<endl;
    // cout<<"ANT COLONY: Number of Edges in Colony[-1]: ";
    // cout<<colony[-1]->pheromone_lines->back()<<endl;//->get_edge_phermone()<<endl;
    // cout<<"ANT COLONY:: Number of Edges in Colony[-1]: "<<colony[-1]->pheromone_lines->size()<<endl;
}

void ACNNTO::print_population() {
    cout << "POPULATIONS: " << endl;
    for (int32_t i = 0; i < (int32_t)population.size(); i++) {
        cout << "\tPOPULATION " << i << ":" << endl;

        cout << "\t" << RNN_Genome::print_statistics_header() << endl;

        cout << "\t" << population[i]->print_statistics() << endl;
    }

    cout << endl << endl;

    if (log_file != NULL) {

        //make sure the log file is still good
        if (!log_file->good()) {
            log_file->close();
            delete log_file;

            string output_file = output_directory + "/fitness_log.csv";
            log_file = new ofstream(output_file, std::ios_base::app);

            if (!log_file->is_open()) {
                cerr << "ERROR, could not open ACNNTO output log: '" << output_file << "'" << endl;
                exit(1);
            }
        }

        RNN_Genome *best_genome = get_best_genome();

        std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
        long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - startClock).count();

        (*log_file) << inserted_genomes
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count() << endl;

        memory_log << inserted_genomes
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count() << endl;
    }
}

void ACNNTO::write_memory_log(string filename) {
    ofstream log_file(filename);
    log_file << memory_log.str();
    log_file.close();
}

void ACNNTO::set_possible_node_types(vector<string> possible_node_type_strings) {
    possible_node_types.clear();

    for (int32_t i = 0; i < possible_node_type_strings.size(); i++) {
        string node_type_s = possible_node_type_strings[i];

        bool found = false;

        for (int32_t j = 0; j < NUMBER_NODE_TYPES; j++) {
            if (NODE_TYPES[j].compare(node_type_s) == 0) {
                found = true;
                possible_node_types.push_back(j);
            }
        }

        if (!found) {
            cerr << "ERROR! unknown node type: '" << node_type_s << "'" << endl;
            exit(1);
        }
    }
}

int32_t ACNNTO::population_contains(RNN_Genome* genome) {
    for (int32_t j = 0; j < (int32_t)population.size(); j++) {
        if (population[j]->equals(genome)) {
            return j;
        }
    }

    return -1;
}

bool ACNNTO::populations_full() const {
    if (population.size() < population_size) return false;
    return true;
}

string ACNNTO::get_output_directory() const {
    return output_directory;
}

RNN_Genome* ACNNTO::get_best_genome() {
  if (population.size() <= 0) {
      return NULL;
  } else {
      return population[0];
  }
}

RNN_Genome* ACNNTO::get_worst_genome() {
  if (population.size() <= 0) {
      return NULL;
  } else {
      return population.back();
  }
}

double ACNNTO::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXALT_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double ACNNTO::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXALT_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}

//this will insert a COPY, original needs to be deleted
bool ACNNTO::insert_genome(RNN_Genome* genome) {
    if (!genome->sanity_check()) {
        cerr << "ERROR, genome failed sanity check on insert!" << endl;
        exit(1);
    }

    double new_fitness = genome->get_fitness();
    bool was_inserted = true;

    inserted_genomes++;
    total_bp_epochs += genome->get_bp_iterations();

    genome->update_generation_map(generated_from_map);

    cout << "genomes evaluated: " << setw(10) << inserted_genomes << ", inserting: " << parse_fitness(genome->get_fitness()) << endl;


    if (population.size() >= population_size  && new_fitness > population.back()->get_fitness()) {
      cout << "ignoring genome, fitness: " << new_fitness << " > worst population" << " fitness: " << population.back()->get_fitness() << endl;
      print_population();
      // old_reward_colony(genome, 0.85);
      reward_colony(genome, false);
      return false;
    }

    int32_t duplicate_genome = population_contains(genome);
    if (duplicate_genome >= 0) {
        //if fitness is better, replace this genome with new one
        cout << "found duplicate at position: " << duplicate_genome << endl;

        RNN_Genome *duplicate = population[duplicate_genome];
        if (duplicate->get_fitness() > new_fitness) {
            //erase the genome with loewr fitness from the vector;
            cout << "REPLACING DUPLICATE GENOME, fitness of genome in search: " << parse_fitness(duplicate->get_fitness()) << ", new fitness: " << parse_fitness(genome->get_fitness()) << endl;
            population.erase(population.begin() + duplicate_genome);
            delete duplicate;
            // old_reward_colony(genome, 1.15);
            reward_colony(genome, true);

        } else {
            cerr << "\tpopulation already contains genome! not inserting." << endl;
            print_population();
            return false;
        }
    }

    if (population.size() < population_size || population.back()->get_fitness() > new_fitness) {
        //this genome will be inserted
        was_inserted = true;
        // old_reward_colony(genome, 1.15);
        reward_colony(genome, true);

        cout<<"1- Population Size: "<<population.size()<< " POPULATION_SIZE: "<<population_size<<endl;
        cout<<"1-- CEHCKING: \n";
        if (population.size() == 0 || genome->get_fitness() < get_best_genome()->get_fitness()) {
            cout<<"2-- CEHCKING: \n";
            cout << "new best fitness!" << endl;

            if (genome->get_fitness() != EXALT_MAX_DOUBLE) {
                //need to set the weights for non-initial genomes so we
                //can generate a proper graphviz file
                vector<double> best_parameters = genome->get_best_parameters();
                genome->set_weights(best_parameters);
            }

            genome->write_graphviz(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".gv");
            genome->write_to_file(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".bin", true);

        }

        cout << "inserting new genome to population " << endl;
        //inorder insert the new individual
        RNN_Genome *copy = genome->copy();
        // cout << "created copy with island: " << copy->get_island() << endl;

        population.insert( upper_bound(population.begin(), population.end(), copy, sort_genomes_by_fitness()), copy);
        cout << "finished insert" << endl;

        //delete the worst individual if we've reached the population size
        if ((int32_t)population.size() > population_size) {
            cout << "deleting worst genome" << endl;
            RNN_Genome *worst = population.back();
            population.pop_back();

            delete worst;
        }
    } else {
        was_inserted = false;
        cout << "not inserting genome due to poor fitness" << endl;
    }

    print_population();

    cout << "printed population!" << endl;

    return was_inserted;
}

void ACNNTO::initialize_genome_parameters(RNN_Genome* genome) {
    genome->set_bp_iterations(bp_iterations);
    genome->set_learning_rate(learning_rate);
    genome->initialize_randomly();

    if (use_high_threshold) genome->enable_high_threshold(high_threshold);
    if (use_low_threshold) genome->enable_low_threshold(low_threshold);
}

EDGE_Pheromone* ACNNTO::pick_line(vector<EDGE_Pheromone*>* edges_pheromones){
  double sum_pheromones = 0;
  for ( int i=0; i<edges_pheromones->size(); i++){
    sum_pheromones+=edges_pheromones->at(i)->get_edge_phermone();
  }
  uniform_real_distribution<double> rng(0.0, 1.0);
  double rand_gen = rng(generator) * sum_pheromones;
  for ( int i = 0; i < edges_pheromones->size(); i++) {
      cout<<"Edge : "<<edges_pheromones->at(i)->get_output_innovation_number()<<" Pheromone Value: "<<edges_pheromones->at(i)->get_edge_phermone()<<endl;
  }
  int a;
  for ( a = 0; a < edges_pheromones->size(); a++) {
      if ( rand_gen<=edges_pheromones->at(a)->get_edge_phermone() )
        break;
      else
        rand_gen-= edges_pheromones->at(a)->get_edge_phermone();
  }

  return edges_pheromones->at(a);
}

int ACNNTO::pick_node_type(double* type_pheromones){
    uniform_real_distribution<double> rng(0.0, 1.0);
    double rand_gen = rng(generator);
    double highest_value = type_pheromones[0]; int a = 0;
    if ( rand_gen < EXPLORATION_FACTOR ){
        for ( int i=0; i< node_types.size(); i++){
            if (highest_value <  type_pheromones[i])
                highest_value = type_pheromones[i];
                a = i;
        }
    }
    else{
        double sum_pheromones = 0;
        for (int i=0; i<node_types.size(); i++){
          sum_pheromones+=type_pheromones[i];
        }
        highest_value =  type_pheromones[0] / sum_pheromones;; a = 0;
        for ( int i=1; i< node_types.size(); i++){
            int dum = type_pheromones[i] / sum_pheromones;
            if (highest_value < dum )
                highest_value = dum;
                a = i;
        }
    }
    cout<<"AA: "<<a<<endl;
    switch (a) {
    case 0: return FEED_FORWARD_NODE;
        break;
    case 1: return UGRNN_NODE;
        break;
    case 2: return MGU_NODE;
        break;
    case 3: return GRU_NODE;
        break;
    case 4: return DELTA_NODE;
        break;
    case 5: return LSTM_NODE;
        break;
    }
    cerr << "ERROR choosing node type! None existing type!!"<<endl;
    return -1;
}


int ACNNTO::old_pick_node_type(double* type_pheromones){
  double sum_pheromones = 0;
  // for (int i=0; i<sizeof(type_pheromones)/sizeof(type_pheromones[0]); i++){
  for (int i=0; i<node_types.size(); i++){
    sum_pheromones+=type_pheromones[i];
  }
  uniform_real_distribution<double> rng(0.0, 1.0);
  double rand_gen = rng(generator) * sum_pheromones;
  // cout << "Node Types Pheromones Sum: " << sum_pheromones <<endl;
  // cout << "rand_gen: " << rand_gen << endl;
  for ( int i=0; i<node_types.size(); i++){
      // cout<<"Node type: "<<i<<" Pheromone Value: "<<type_pheromones[i]<<endl;
  }
  int a;
  for ( a = 0; a < node_types.size(); a++) {
      if ( rand_gen<=type_pheromones[a] )
        break;
      else
        rand_gen-= type_pheromones[a];
  }
  // cout<<"Ant Choice: "<<a<<endl;
  // cout<<"IN PICK NODE TYPE\n";
  // getchar();
  switch (a) {
  case 0: return FEED_FORWARD_NODE;
      break;
  case 1: return UGRNN_NODE;
      break;
  case 2: return MGU_NODE;
      break;
  case 3: return GRU_NODE;
      break;
  case 4: return DELTA_NODE;
      break;
  case 5: return LSTM_NODE;
      break;
  }
  cerr << "ERROR choosing node type! None existing type!!"<<endl;
  return -1;
}

/*
  If genome fitness is good compared to the population=>
      reward it and increase the pheromones on its active nodes and edges
  If genome fitness is bad=>
      penalize it and reduce the pheromones on its active nodes and edges
*/

void ACNNTO::reward_colony(RNN_Genome* g, bool reward){
    cout<<"REWARDING COLONY!"<<endl;
    double max_pheromone = 1/( PHEROMONE_DECAY_PARAMETER * g->get_fitness() );
    double min_pheromone = max_pheromone / ( 2 * g->nodes.size() * node_types.size() );
    for ( int i=0; i<g->nodes.size(); i++){
      if ( g->nodes[i]->enabled==true ){
        int32_t node_inno = g->nodes[i]->get_innovation_number();
        int32_t node_type = g->nodes[i]->node_type;
        if (i==g->nodes.size()-1)
            node_inno*=-1;

        double pheromone_update;
        if ( reward )
            pheromone_update = ( 1 - PHEROMONE_DECAY_PARAMETER ) * colony[node_inno]->type_pheromones[node_types[node_type]] + PHEROMONE_DECAY_PARAMETER * ( 1 / g->get_fitness() ) ;
        else
            pheromone_update = ( 1 - PHEROMONE_UPDATE_STRENGTH ) * colony[node_inno]->type_pheromones[node_types[node_type]] + PHEROMONE_UPDATE_STRENGTH ;

        if ( pheromone_update > max_pheromone )
            colony[node_inno]->type_pheromones[node_types[node_type]] = max_pheromone;
        else if ( pheromone_update < min_pheromone )
            colony[node_inno]->type_pheromones[node_types[node_type]] = min_pheromone;
        else
            colony[node_inno]->type_pheromones[node_types[node_type]] = pheromone_update;

        if (g->nodes[i]->layer_type==0){
            for (int m=0; m<colony[-1]->pheromone_lines->size(); m++){
                if ( colony[-1]->pheromone_lines->at(m)->get_output_innovation_number() == g->nodes[i]->innovation_number){
                    if ( reward )
                        colony[-1]->pheromone_lines->at(m)->edge_pheromone = ( 1 - PHEROMONE_DECAY_PARAMETER ) * colony[-1]->pheromone_lines->at(m)->edge_pheromone + PHEROMONE_DECAY_PARAMETER * ( 1 / g->get_fitness() ) ;
                    else
                        colony[-1]->pheromone_lines->at(m)->edge_pheromone = ( 1 - PHEROMONE_UPDATE_STRENGTH ) * colony[-1]->pheromone_lines->at(m)->edge_pheromone + PHEROMONE_UPDATE_STRENGTH ;
                }
            }
        }
      }
  }


  min_pheromone = max_pheromone / ( 2 * g->edges.size() );
  for ( int i=0; i<g->edges.size(); i++){
        if (g->edges[i]->enabled){
            int32_t edge_inno = g->edges[i]->get_innovation_number();
            for ( int j=0; j<colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++){
                if ( edge_inno == colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ){
                    double pheromone_update;
                    if (reward)
                        pheromone_update = (1 - PHEROMONE_DECAY_PARAMETER) * colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone + PHEROMONE_DECAY_PARAMETER * ( 1 / g->get_fitness() );
                    else
                        pheromone_update = (1 - PHEROMONE_UPDATE_STRENGTH) * colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone + PHEROMONE_UPDATE_STRENGTH  ;
                    if ( pheromone_update > max_pheromone )
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = max_pheromone ;
                    else if ( pheromone_update < min_pheromone  )
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = min_pheromone ;
                    else
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = pheromone_update ;
                }
            }
        }
    }
}


void ACNNTO::old_reward_colony(RNN_Genome* g, double treat_pheromone){
    if (treat_pheromone==1.15){
        for ( int i=0; i<g->nodes.size(); i++){
            if (g->nodes[i]->enabled==true){
                int32_t node_inno = g->nodes[i]->get_innovation_number();
                int32_t node_type = g->nodes[i]->node_type;
                if (i==g->nodes.size()-1)
                    node_inno*=-1;

                if (colony[node_inno]->type_pheromones[node_types[node_type]]*treat_pheromone < PHEROMONES_THERESHOLD )
                    colony[node_inno]->type_pheromones[node_types[node_type]]*=treat_pheromone;
                else
                    colony[node_inno]->type_pheromones[node_types[node_type]] = PHEROMONES_THERESHOLD;
            }
        }
    }
  if (treat_pheromone==0.85){
    for ( int i=0; i<g->nodes.size(); i++){
        cout<<g->nodes.size()<<endl;
        if (g->nodes[i]->enabled==true){
            int32_t node_inno = g->nodes[i]->get_innovation_number();
            if (i==g->nodes.size()-1)
                node_inno*=-1;
            int32_t node_type = g->nodes[i]->node_type;
            colony[node_inno]->type_pheromones[node_types[node_type]]*=treat_pheromone;
      }
    }
  }

  for ( int i=0; i<g->edges.size(); i++){
        if (g->edges[i]->enabled){
            int32_t edge_inno = g->edges[i]->get_innovation_number();
            for ( int j=0; j<colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++){
                if ( edge_inno == colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ){
                    if (colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone*treat_pheromone < PHEROMONES_THERESHOLD){
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone*=treat_pheromone;
                        break;
                    }
                    colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = PHEROMONES_THERESHOLD;
                    break;
                }
            }
        }
    }
}

/*Will reduce phermones periodically*/
void ACNNTO::evaporate_pheromones(){
    cout<<"PROBLEM IS BELOW!\n";
    for ( int i=-1; i<colony.size(); i++){
        if(i==colony.size()-1)
            i*=-1;
        colony[i]->type_pheromones[0]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
        colony[i]->type_pheromones[1]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
        colony[i]->type_pheromones[2]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
        colony[i]->type_pheromones[3]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
        colony[i]->type_pheromones[4]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
        for ( int j=0; j<colony[i]->pheromone_lines->size(); j++){
            colony[i]->pheromone_lines->at(j)->edge_pheromone*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
        }
    }
    cout<<"PROBLEM IS ABOVE!\n";
}


RNN_Node* ACNNTO::check_node_existance(vector<RNN_Node_Interface*> &rnn_nodes,   EDGE_Pheromone* pheromone_line){
  for ( int i=0; i<rnn_nodes.size(); i++){
    if (rnn_nodes[i]->get_innovation_number() == abs(pheromone_line->get_output_innovation_number())){
      return (RNN_Node*)rnn_nodes[i];
    }
  }
  // rnn_nodes.push_back(new RNN_Node(
  //                                   abs(pheromone_line->get_output_innovation_number()),
  //                                   colony[pheromone_line->get_output_innovation_number()]->get_layer_type(),
  //                                   colony[pheromone_line->get_output_innovation_number()]->get_current_layer(),
  //                                   old_pick_node_type(colony[pheromone_line->get_output_innovation_number()]->type_pheromones)
  //                               ));



    int innovation_number   = abs(pheromone_line->get_output_innovation_number());
    int layer_type          = colony[pheromone_line->get_output_innovation_number()]->get_layer_type();
    double current_layer    = colony[pheromone_line->get_output_innovation_number()]->get_current_layer();
    int node_type           = old_pick_node_type(colony[pheromone_line->get_output_innovation_number()]->type_pheromones);

    // rnn_nodes.push_back(new RNN_Node(innovation_number, layer_type, current_layer, node_type));

                                    if (node_type == LSTM_NODE) {
                                        rnn_nodes.push_back( new LSTM_Node(innovation_number, layer_type, current_layer) );
                                    } else if (node_type == DELTA_NODE) {
                                        rnn_nodes.push_back( new Delta_Node(innovation_number, layer_type, current_layer) );
                                    } else if (node_type == GRU_NODE) {
                                        rnn_nodes.push_back( new GRU_Node(innovation_number, layer_type, current_layer) );
                                    } else if (node_type == MGU_NODE) {
                                        rnn_nodes.push_back( new MGU_Node(innovation_number, layer_type, current_layer) );
                                    } else if (node_type == UGRNN_NODE) {
                                        rnn_nodes.push_back( new UGRNN_Node(innovation_number, layer_type, current_layer) );
                                    } else if (node_type == FEED_FORWARD_NODE ) {
                                        rnn_nodes.push_back( new RNN_Node(innovation_number, layer_type, current_layer, node_type) );
                                    } else {
                                        cerr << "ACNNTO:: Error reading node from stream, unknown node_type: " << node_type << endl;
                                        exit(1);
                                    }

  return (RNN_Node*)rnn_nodes.back();
}
void ACNNTO::check_edge_existance(vector<RNN_Edge*> &rnn_edges, int32_t innovation_number, RNN_Node* in_node, RNN_Node* out_node){
  for ( int i=0; i<rnn_edges.size(); i++){
    if (rnn_edges[i]->get_innovation_number() == innovation_number){
      return;
    }
  }
  rnn_edges.push_back(new RNN_Edge(innovation_number, in_node, out_node));
  return;
}
void ACNNTO::check_recurrent_edge_existance(vector<RNN_Recurrent_Edge*> &recurrent_edges, int32_t innovation_number, int32_t depth, RNN_Node* in_node, RNN_Node* out_node){
  for ( int i=0; i<recurrent_edges.size(); i++){
    if (recurrent_edges[i]->get_innovation_number() == innovation_number){
      return;
    }
  }
  recurrent_edges.push_back(new RNN_Recurrent_Edge(innovation_number, depth, in_node, out_node));
  return;
}

void ACNNTO::prepare_new_genome(RNN_Genome* genome){
  int32_t island  = -1;
  genome->set_island(island);
  genome->set_parameter_names(input_parameter_names, output_parameter_names);
  genome->set_normalize_bounds(normalize_mins, normalize_maxs);

  edge_innovation_count = genome->edges.size() + genome->recurrent_edges.size();
  node_innovation_count = genome->nodes.size();

  genome->set_generated_by("initial");
  initialize_genome_parameters(genome);
  genome->initialize_randomly();
  genome->best_validation_mse = EXALT_MAX_DOUBLE;
  genome->best_validation_mae = EXALT_MAX_DOUBLE;
  genome->best_parameters.clear();
  // insert_genome(genome->copy());
  //genome->clear_generated_by();

  // genome->set_generation_id(generated_genomes);
}

/* Each ant will marsh and choose a path and will turn each edge/node active in its path */
RNN_Genome* ACNNTO::ants_march(){
    if (generated_genomes == max_genomes){
        return NULL;
    }
    for (auto const& x: colony){
        cout<<"NoDe Number: "<<x.first
        <<" Layer Type: "<<colony[x.first]->get_layer_type()
        <<" Current Layer: "<<colony[x.first]->get_current_layer()
        <<endl;
    }
  vector<RNN_Node_Interface*> rnn_nodes;
  vector<RNN_Edge*> rnn_edges;
  vector<RNN_Recurrent_Edge*> recurrent_edges;
  EDGE_Pheromone* pheromone_line = NULL;
  while (1){
    rnn_nodes.clear();
    rnn_edges.clear();
    recurrent_edges.clear();
    for (int ant=0; ant<ANTS; ant++){
        cout<<"ANT:: "<<ant<<endl;
        if (generated_genomes!=0 && generated_genomes%100==0){
          evaporate_pheromones();
}
      pheromone_line = pick_line(colony[-1]->pheromone_lines); //begine with imaginary node
      // RNN_Node* node = new RNN_Node(abs(pheromone_line->get_output_innovation_number()),
      //                                   colony[pheromone_line->get_output_innovation_number()]->get_layer_type(),
      //                                   colony[pheromone_line->get_output_innovation_number()]->get_current_layer(),
      //                                   pick_node_type(colony[pheromone_line->get_output_innovation_number()]->type_pheromones)
      //                                 );
      RNN_Node* current_node = check_node_existance(rnn_nodes,  pheromone_line);

      RNN_Node* previous_node = current_node;
      cout<<"FIRST NODE: "<<previous_node->get_innovation_number()<<endl;
      cout<<"\t************************************************\n";
      cout<<"\t************************************************\n";
      cout<<"\t************************************************\n";
      while (1){
        cout<<"\tNode Innovation #: "<<pheromone_line->get_output_innovation_number()<<endl;
        cout<<"\tLayer Type       : "<<colony[pheromone_line->get_output_innovation_number()]->get_layer_type()<<endl;
        // cout<<"\tCurrent Layer    : "<<colony[pheromone_line->get_output_innovation_number()]->get_current_layer()<<endl;
        pheromone_line = pick_line(colony[pheromone_line->get_output_innovation_number()]->pheromone_lines);
        cout<<"\t**NODES**\n";
        // for (int y=0; y<rnn_nodes.size(); y++)
        //   cout<<"\t\tNODES["<<y<<"]: "<<rnn_nodes[y]->get_innovation_number()<<" Layer Type: "<<rnn_nodes[y]->layer_type<< " Node Address: "<< rnn_nodes[y]<<endl;
        cout<<"\tNumber of Nodes: "<<rnn_nodes.size()<<endl;
        // RNN_Node* current_node = new RNN_Node(abs(pheromone_line->get_output_innovation_number()),
        //                                   colony[pheromone_line->get_output_innovation_number()]->get_layer_type(),
        //                                   colony[pheromone_line->get_output_innovation_number()]->get_current_layer(),
        //                                   pick_node_type(colony[pheromone_line->get_output_innovation_number()]->type_pheromones));
        // current_node = check_node_existance(rnn_nodes, current_node);
        current_node = check_node_existance(rnn_nodes, pheromone_line);




        cout<<"\tEdge Depth: "<<pheromone_line->get_depth()<<endl;
        if (pheromone_line->get_depth()==0 && previous_node->get_innovation_number()<current_node->get_innovation_number()){
          check_edge_existance(rnn_edges, pheromone_line->get_edge_innovation_number(), previous_node, current_node);
          cout<<"\t**EDGES**\n";
          // for (int y=0; y<rnn_edges.size(); y++)
          //   cout<<"\t\tEDGE["<<y<<"] (innov: "<<rnn_edges[y]->innovation_number<<")  From NODE: "<<rnn_edges[y]->get_input_innovation_number()<<" -- To NODE: "<<rnn_edges[y]->get_output_innovation_number()<<endl;
        }
        else{
          check_recurrent_edge_existance(recurrent_edges, pheromone_line->get_edge_innovation_number(), pheromone_line->get_depth(), previous_node, current_node);
          cout<<"\t**RecEDGES**\n";
          // for (int y=0; y<recurrent_edges.size(); y++)
          //   cout<<"\t\tRecEDGE["<<y<<"] (inov: "<<recurrent_edges[y]->innovation_number<<")  From NODE: "<<recurrent_edges[y]->get_input_innovation_number()<<" -- To NODE: "<<recurrent_edges[y]->get_output_innovation_number()<<" Depth: "<<recurrent_edges[y]->recurrent_depth<<endl;
        }
        cout<<"\t########################################\n";
        previous_node = current_node;
        if (pheromone_line->get_output_innovation_number()<0)
          break;  //break while() when its iterator reaches an output node
      }
    }

    for (int y=0; y<rnn_nodes.size(); y++)
      cout<<"\t\tNODES["<<y<<"]: "<<rnn_nodes[y]->get_innovation_number()<<" Layer Type: "<<rnn_nodes[y]->layer_type<< " Node Address: "<< rnn_nodes[y] <<
                "Node Type: " << rnn_nodes[y]->node_type <<endl;
    for (int y=0; y<rnn_edges.size(); y++)
      cout<<"\t\tEDGE["<<y<<"] (innov: "<<rnn_edges[y]->innovation_number<<" Add: "<< rnn_edges[y] << ")  From NODE: "<<rnn_edges[y]->get_input_innovation_number()<< " (" << rnn_edges[y]->input_node << ") " <<" -- To NODE: "<<rnn_edges[y]->get_output_innovation_number()<< " (" << rnn_edges[y]->output_node << ") "<<endl;
    for (int y=0; y<recurrent_edges.size(); y++)
      cout<<"\t\tRecEDGE["<<y<<"] (inov: "<<recurrent_edges[y]->innovation_number<<" Add: " << recurrent_edges[y] <<  ")  From NODE: "<<recurrent_edges[y]->get_input_innovation_number() << " (" << recurrent_edges[y]->output_node << ") "<< " -- To NODE: "<<recurrent_edges[y]->get_output_innovation_number() << " (" << recurrent_edges[y]->output_node << ") "<< " Depth: "<<recurrent_edges[y]->recurrent_depth<<endl;

    for (int i=0; i<this->number_inputs; i++){
      bool add_node = true;
      for (int j=0; j<rnn_nodes.size(); j++){
        if (rnn_nodes[j]->layer_type==INPUT_LAYER && rnn_nodes[j]->innovation_number==i){
          add_node = false;
          break;
        }
      }
      if (add_node){
        RNN_Node* node = new RNN_Node(-(i+100), INPUT_LAYER, 0, FEED_FORWARD_NODE);
        node->enabled = false;
        rnn_nodes.push_back(node);
      }
    }
    for (int i=0; i<rnn_nodes.size(); i++){
      cout<<"Node Innov No: "<<rnn_nodes[i]->innovation_number<< " Enabled: "<<rnn_nodes[i]->enabled<<endl;
    }

    // cout<<"NUMBER of Nodes: "<<rnn_nodes.size()<<endl;
    RNN_Genome* g = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
    g->write_graphviz("genome.gv");
    // cout<<"PRESS ENTER TO CONTINUE...\n";
    // getchar();
    if (!g->outputs_unreachable()){
      g->set_generation_id(generated_genomes++);
      prepare_new_genome(g);
      cout<<"ANTS SUCCEEDED IN FINDING A COMPLETE NN STRUCTURE.... WILL BEGIN GENOME EVALUATION..."<<endl;
      return g;
    }
    cout<<"ANTS FAILED TO FIND A COMPLETE NN STRUCTURE.... BEGINING ANOTHER ITERATION..."<<endl;
  }
}
