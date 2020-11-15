#include <chrono>

#include <condition_variable>
using std::condition_variable;

#include <iomanip>
using std::setw;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "rnn/rnn_genome.hxx"

#include "time_series/time_series.hxx"
#include "common/weight_initialize.hxx"

vector<string> arguments;

vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    TimeSeriesSets *time_series_sets = NULL;
    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);
    RNN_Genome *genome = new RNN_Genome(genome_filename);

    int32_t bp_iterations;
    get_argument(arguments, "--bp_iterations", true, bp_iterations);

    vector<string> test_filenames;
    get_argument_vector(arguments, "--test_filenames", true, test_filenames);

    string weight_initialize_string = "xavier";
    get_argument(arguments, "--weight_initialize", false, weight_initialize_string);
    WeightType weight_initialize;
    weight_initialize = get_enum_from_string(weight_initialize_string);


    genome->set_bp_iterations(bp_iterations);
    genome->set_weight_initialize(weight_initialize);
    genome->initialize_randomly();
    genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);

    vector<double> best_parameters = genome->get_best_parameters();
    
    Log::info("MSE: %lf\n", genome->get_mse(best_parameters, validation_inputs, validation_outputs));
    Log::info("MAE: %lf\n", genome->get_mae(best_parameters, validation_inputs, validation_outputs));

    Log::release_id("main");
    return 0;
}
