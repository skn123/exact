#ifndef RNN_GENERATE_NN_HXX
#define RNN_GENERATE_NN_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <random>
using std::minstd_rand0;
using std::uniform_int_distribution;

#include "rnn/rnn_genome.hxx"
#include "rnn/recurrent_depth.hxx"
#include "common/weight_initialize.hxx"

RNN_Genome* create_ff(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize, WeightType weight_inheritance, WeightType mutated_component_weight);

RNN_Genome* create_jordan(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize);

RNN_Genome* create_elman(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, int max_recurrent_depth, WeightType weight_initialize);

RNN_Genome* create_lstm(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, WeightType weight_initialize, Recurrent_Depth *node_rec_depth);

RNN_Genome* create_delta(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, WeightType weight_initialize, Recurrent_Depth *node_rec_depth);

RNN_Genome* create_gru(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, WeightType weight_initialize, Recurrent_Depth *node_rec_depth);

RNN_Genome* create_mgu(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, WeightType weight_initialize, Recurrent_Depth *node_rec_depth);

RNN_Genome* create_ugrnn(const vector<string> &input_parameter_names, int number_hidden_layers, int number_hidden_nodes, const vector<string> &output_parameter_names, WeightType weight_initialize, Recurrent_Depth *node_rec_depth);

#endif
