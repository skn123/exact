#include <chrono>

#include <iomanip>
using std::setw;
using std::fixed;
using std::setprecision;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;


#include "mpi.h"

#include "common/arguments.hxx"
#include "common/files.hxx"
#include "common/log.hxx"

#include "rnn/examm.hxx"

#include "time_series/time_series.hxx"

#define WORK_REQUEST_TAG 1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG 3
#define TERMINATE_TAG 4

mutex examm_mutex;

vector<string> arguments;

EXAMM *examm;

vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;

int32_t global_slice;
int32_t global_repeat;

void send_work_request(int target) {
    int work_request_message[1];
    work_request_message[0] = 0;
    MPI_Send(work_request_message, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);
}

void receive_work_request(int source) {
    MPI_Status status;
    int work_request_message[1];
    MPI_Recv(work_request_message, 1, MPI_INT, source, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);
}

RNN_Genome* receive_genome_from(int source) {
    MPI_Status status;
    int length_message[1];
    MPI_Recv(length_message, 1, MPI_INT, source, GENOME_LENGTH_TAG, MPI_COMM_WORLD, &status);

    int length = length_message[0];

    Log::debug("receiving genome of length: %d from: %d\n", length, source);

    char* genome_str = new char[length + 1];

    Log::debug("receiving genome from: %d\n", source);
    MPI_Recv(genome_str, length, MPI_CHAR, source, GENOME_TAG, MPI_COMM_WORLD, &status);

    genome_str[length] = '\0';

    Log::trace("genome_str:\n%s\n", genome_str);

    RNN_Genome* genome = new RNN_Genome(genome_str, length);

    delete [] genome_str;
    return genome;
}

void send_genome_to(int target, RNN_Genome* genome) {
    char *byte_array;
    int32_t length;

    genome->write_to_array(&byte_array, length);

    Log::debug("sending genome of length: %d to: %d\n", length, target);

    int length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, GENOME_LENGTH_TAG, MPI_COMM_WORLD);

    Log::debug("sending genome to: %d\n", target);
    MPI_Send(byte_array, length, MPI_CHAR, target, GENOME_TAG, MPI_COMM_WORLD);

    free(byte_array);
}

void send_terminate_message(int target) {
    int terminate_message[1];
    terminate_message[0] = 0;
    MPI_Send(terminate_message, 1, MPI_INT, target, TERMINATE_TAG, MPI_COMM_WORLD);
}

void receive_terminate_message(int source) {
    MPI_Status status;
    int terminate_message[1];
    MPI_Recv(terminate_message, 1, MPI_INT, source, TERMINATE_TAG, MPI_COMM_WORLD, &status);
}

void master(int max_rank) {
    int terminates_sent = 0;

    while (true) {
        //wait for a incoming message
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);

        //if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            receive_work_request(source);

            examm_mutex.lock();
            RNN_Genome *genome = examm->generate_genome();
            examm_mutex.unlock();

            if (genome == NULL) { //search was completed if it returns NULL for an individual
                //send terminate message
                Log::debug("terminating worker: %d\n", source);
                send_terminate_message(source);
                terminates_sent++;

                Log::debug("sent: %d terminates of %d\n", terminates_sent, (max_rank - 1));
                if (terminates_sent >= max_rank - 1) return;

            } else {
                //genome->write_to_file( examm->get_output_directory() + "/before_send_gen_" + to_string(genome->get_generation_id()) );

                //send genome
                Log::debug("sending genome to: %d\n", source);
                send_genome_to(source, genome);

                //delete this genome as it will not be used again
                delete genome;
            }
        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome from: %d\n", source);
            RNN_Genome *genome = receive_genome_from(source);

            examm_mutex.lock();
            examm->insert_genome(genome);
            examm_mutex.unlock();

            delete genome;
            //this genome will be deleted if/when removed from population
        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d\n", source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void worker(int rank) {
    string worker_id = "worker_slice_" + to_string(global_slice) + "_repeat_" + to_string(global_repeat) + "_" + to_string(rank);
    Log::set_id(worker_id);

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request(0);
        Log::debug("sent work request!\n");

        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int tag = status.MPI_TAG;

        Log::debug("probe received message with tag: %d\n", tag);

        if (tag == TERMINATE_TAG) {
            Log::debug("received terminate tag!\n");
            receive_terminate_message(0);
            break;

        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome!\n");
            RNN_Genome* genome = receive_genome_from(0);

            string log_id = "slice_" + to_string(global_slice) + "_repeat_" + to_string(global_repeat) + "_genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
            Log::set_id(log_id);
            genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
            Log::release_id(log_id);

            //go back to the worker's log for MPI communication
            Log::set_id(worker_id);

            send_genome_to(0, genome);

            delete genome;
        } else {
            Log::fatal("ERROR: received message with unknown tag: %d\n", tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    //release the log file for the worker communication
    Log::release_id(worker_id);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + to_string(rank));
    Log::restrict_to_rank(0);

#include "rnn/argparse.cxx"

    //only have the master process print TSS info
    if (rank == 0 && argument_exists(arguments, "--write_time_series")) {
        string base_filename;
        get_argument(arguments, "--write_time_series", true, base_filename);
        time_series_sets->write_time_series_sets(base_filename);
    }

    int32_t repeats;
    get_argument(arguments, "--repeats", true, repeats);

    int fold_size = 2;
    get_argument(arguments, "--fold_size", true, fold_size);

    Log::clear_rank_restriction();

    for (int32_t i = 0; i < time_series_sets->get_number_series(); i += fold_size) {
        vector<int> training_indexes;
        vector<int> test_indexes;

        for (uint32_t j = 0; j < time_series_sets->get_number_series(); j += fold_size) {
            if (j == i) {
                for (int k = 0; k < fold_size; k++) {
                    test_indexes.push_back(j + k);
                }
            } else {
                for (int k = 0; k < fold_size; k++) {
                    training_indexes.push_back(j + k);
                }
            }
        }

        time_series_sets->set_training_indexes(training_indexes);
        time_series_sets->set_test_indexes(test_indexes);

        //time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
        //time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

        string slice_output_directory = output_directory + "/slice_" + to_string(i);
        mkpath(slice_output_directory.c_str(), 0777);
        ofstream slice_times_file(output_directory + "/slice_" + to_string(i) + "_runtimes.csv");

        for (int k = 0; k < repeats; k++) {
            string current_output_directory = slice_output_directory + "/repeat_" + to_string(k);
            mkpath(current_output_directory.c_str(), 0777);

            //set to the master/workers can specify the right log id
            global_slice = i;
            global_repeat = k;

            if (rank == 0) {
                string examm_log_id = "examm_slice_" + to_string(global_slice) + "_repeat_" + to_string(global_repeat);
                Log::set_id(examm_log_id);

                examm = new EXAMM(population_size, number_islands, max_genomes, extinction_event_generation_number, islands_to_exterminate, island_ranking_method,
                    repopulation_method, repopulation_mutations, repeat_extinction,
                    speciation_method,
                    species_threshold, fitness_threshold,
                    neat_c1, neat_c2, neat_c3,
                    time_series_sets->get_input_parameter_names(),
                    time_series_sets->get_output_parameter_names(),
                    time_series_sets->get_normalize_type(),
                    time_series_sets->get_normalize_mins(),
                    time_series_sets->get_normalize_maxs(),
                    time_series_sets->get_normalize_avgs(),
                    time_series_sets->get_normalize_std_devs(),
                    weight_initialize, weight_inheritance, mutated_component_weight,
                    bp_iterations, learning_rate,
                    use_high_threshold, high_threshold,
                    use_low_threshold, low_threshold,
                    use_dropout, dropout_probability,
                    min_recurrent_depth, max_recurrent_depth,
                    use_regression,
                    output_directory,
                    seed_genome,
                    start_filled);

                examm->set_possible_node_types(possible_node_types);

                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
                master(max_rank);
                std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
                long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                //examm->write_memory_log(current_output_directory + "/memory_fitness_log.csv");

                slice_times_file << milliseconds << endl;

                RNN_Genome *best_genome = examm->get_best_genome();

                string binary_file = slice_output_directory + "/repeat_best_" + to_string(k) + ".bin";
                string graphviz_file = slice_output_directory + "/repeat_best_" + to_string(k) + ".gv";

                Log::debug("writing best genome to '%s' and '%s'\n", binary_file.c_str(), graphviz_file.c_str());
                best_genome->write_to_file(binary_file);
                best_genome->write_graphviz(graphviz_file);

                delete examm;
                Log::release_id(examm_log_id);
            } else {
                worker(rank);
            }
            Log::set_id("main_" + to_string(rank));

            MPI_Barrier(MPI_COMM_WORLD);
            Log::debug("rank %d completed slice %d of %d repeat %d of %d\n", rank, i, time_series_sets->get_number_series(), k, repeats);
        }

        slice_times_file.close();
    }

    MPI_Finalize();
    Log::release_id("main_" + to_string(rank));

    return 0;
}
