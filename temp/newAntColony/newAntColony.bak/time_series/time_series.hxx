#ifndef EXALT_TIME_SERIES_HXX
#define EXALT_TIME_SERIES_HXX

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <map>
using std::map;

#include <vector>
using std::vector;

class TimeSeries {
    private:
        string name;

        double min;
        double average;
        double max;
        double std_dev;
        double variance;
        double min_change;
        double max_change;

        vector<double> values;

        TimeSeries();
    public:
        TimeSeries(string _name);

        void add_value(double value);
        double get_value(int i);

        void calculate_statistics();
        void print_statistics(ostream &out);

        int get_number_values() const;

        double get_min() const;
        double get_average() const;
        double get_max() const;
        double get_std_dev() const;
        double get_variance() const;
        double get_min_change() const;
        double get_max_change() const;

        void normalize_min_max(double min, double max, bool verbose = false);

        void cut(int32_t start, int32_t stop);

        TimeSeries* copy();
};

class TimeSeriesSet {
    private:
        int number_rows;
        string filename;

        vector<string> fields;

        map<string, TimeSeries*> time_series;

        TimeSeriesSet();
    public:


        TimeSeriesSet(string _filename, const vector<string> &_fields);

        void add_time_series(string name);

        int get_number_rows() const;
        int get_number_columns() const;
        string get_filename() const;

        vector<string> get_fields() const;

        double get_min(string field);
        double get_average(string field);
        double get_max(string field);
        double get_std_dev(string field);
        double get_variance(string field);
        double get_min_change(string field);
        double get_max_change(string field);

        void normalize_min_max(string field, double min, double max, bool verbose = false);

        void export_time_series(vector< vector<double> > &data);
        void export_time_series(vector< vector<double> > &data, const vector<string> &requested_fields);
        void export_time_series(vector< vector<double> > &data, const vector<string> &requested_fields, int32_t time_offset);

        TimeSeriesSet* copy();

        void cut(int32_t start, int32_t stop);
        void split(int slices, vector<TimeSeriesSet*> &sub_series);

        void select_parameters(const vector<string> &parameter_names);
        void select_parameters(const vector<string> &input_parameter_names, const vector<string> &output_parameter_names);
};

class TimeSeriesSets {
    private:
        bool normalized;

        vector<string> filenames;

        vector<int> training_indexes;
        vector<int> test_indexes;

        vector<string> input_parameter_names;
        vector<string> output_parameter_names;
        vector<string> all_parameter_names;

        vector<TimeSeriesSet*> time_series;

        map<string,double> normalize_mins;
        map<string,double> normalize_maxs;

        void parse_parameters_string(const vector<string> &p);
        void load_time_series(bool verbose = false);

    public:
        static void help_message();

        TimeSeriesSets();

        static TimeSeriesSets* generate_from_arguments(const vector<string> &arguments, bool verbose = true);
        static TimeSeriesSets* generate_test(const vector<string> &_test_filenames, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names, bool verbose = true);

        void normalize(bool verbose = true);
        void normalize(const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs, bool verbose = true);

        void split_series(int series, int number_slices);
        void split_all(int number_slices);

        void write_time_series_sets(string base_filename);

        void export_time_series(const vector<int> &series_indexes, int time_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs);

        void export_training_series(int time_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs);

        void export_test_series(int time_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs);

        map<string,double> get_normalize_mins() const;
        map<string,double> get_normalize_maxs() const;

        vector<string> get_input_parameter_names() const;
        vector<string> get_output_parameter_names() const;

        int get_number_series() const;

        int get_number_inputs() const;
        int get_number_outputs() const;

        void set_training_indexes(const vector<int> &_training_indexes);
        void set_test_indexes(const vector<int> &_test_indexes);
};

#endif
