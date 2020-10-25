#ifndef EXALT_RNN_NODE_INTERFACE_HXX
#define EXALT_RNN_NODE_INTERFACE_HXX

#include <cstdint>

#include <fstream>
using std::ostream;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/random.hxx"

class RNN;

#define INPUT_LAYER 0
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2

extern const int32_t NUMBER_NODE_TYPES;
extern const string NODE_TYPES[];

#define FEED_FORWARD_NODE 0
#define JORDAN_NODE 1
#define ELMAN_NODE 2
#define UGRNN_NODE 3
#define MGU_NODE 4
#define GRU_NODE 5
#define DELTA_NODE 6
#define LSTM_NODE 7


double sigmoid(double value);
double sigmoid_derivative(double value);
double tanh_derivative(double value);

double bound(double value);

class RNN_Node_Interface {
    protected:
        int32_t innovation_number;
        int32_t layer_type;
        int32_t node_type;

        double depth;

        bool enabled;
        bool backward_reachable;
        bool forward_reachable;

        int32_t series_length;

        vector<double> input_values;
        vector<double> output_values;
        vector<double> error_values;
        vector<double> d_input;

        vector<int32_t> inputs_fired;
        vector<int32_t> outputs_fired;
        int32_t total_inputs;
        int32_t total_outputs;
    public:
        RNN_Node_Interface(int32_t _innovation_number, int32_t _layer_type, double _depth);

        // RNN_Node_Interface(const RNN_Node_Interface& _rnn_node);

        virtual ~RNN_Node_Interface();

        virtual void initialize_randomly(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) = 0;

        virtual void input_fired(int32_t time, double incoming_output) = 0;
        virtual void output_fired(int32_t time, double delta) = 0;
        virtual void error_fired(int32_t time, double error) = 0;

        virtual uint32_t get_number_weights() const = 0;

        virtual void get_weights(vector<double> &parameters) const = 0;
        virtual void set_weights(const vector<double> &parameters) = 0;
        virtual void get_weights(uint32_t &offset, vector<double> &parameters) const = 0;
        virtual void set_weights(uint32_t &offset, const vector<double> &parameters) = 0;
        virtual void reset(int32_t _series_length) = 0;

        virtual void get_gradients(vector<double> &gradients) = 0;

        virtual RNN_Node_Interface* copy() const = 0;

        void write_to_stream(ostream &out);

        int32_t get_layer_type() const;
        int32_t get_innovation_number() const;
        double get_depth() const;
        bool equals(RNN_Node_Interface *other) const;

        bool is_reachable() const;

        friend class RNN_Edge;
        friend class RNN_Recurrent_Edge;
        friend class RNN;
        friend class RNN_Genome;
        friend class ACNNTO;

        friend void get_mse(RNN* genome, const vector< vector<double> > &expected, double &mse, vector< vector<double> > &deltas);
        friend void get_mae(RNN* genome, const vector< vector<double> > &expected, double &mae, vector< vector<double> > &deltas);
};


struct sort_RNN_Nodes_by_innovation {
    bool operator()(RNN_Node_Interface *n1, RNN_Node_Interface *n2) {
        return n1->get_innovation_number() < n2->get_innovation_number();
    }
};


struct sort_RNN_Nodes_by_depth {
    bool operator()(RNN_Node_Interface *n1, RNN_Node_Interface *n2) {
        return n1->get_depth() < n2->get_depth();
    }
};

#endif
