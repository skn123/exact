#include <algorithm>
using std::sort;
using std::upper_bound;

#include <iostream>
using std::ostream;
using std::istream;

#include <limits>
using std::numeric_limits;

#include <random>
using std::mt19937;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

#include <sstream>
using std::istringstream;
using std::ostringstream;

#include <string>
using std::to_string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"
#include "cnn_genome.hxx"
#include "exact.hxx"

EXACT::EXACT(const Images &images, int _population_size, int _epochs) {
    epochs = _epochs;

    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned seed = 10;

    generator = mt19937(seed);
    rng_long = uniform_int_distribution<long>(-numeric_limits<long>::max(), numeric_limits<long>::max());
    rng_double = uniform_real_distribution<double>(0, 1.0);

    node_innovation_count = 0;
    edge_innovation_count = 0;

    population_size = _population_size;

    int total_weights = 0;
    int rows = images.get_image_rows();
    int cols = images.get_image_cols();

    CNN_Node *input_node = new CNN_Node(node_innovation_count, 0, rows, cols, INPUT_NODE);
    node_innovation_count++;
    all_nodes.push_back(input_node);

    for (uint32_t i = 0; i < images.get_number_classes(); i++) {
        CNN_Node *softmax_node = new CNN_Node(node_innovation_count, 1, 1, 1, SOFTMAX_NODE);
        node_innovation_count++;
        all_nodes.push_back(softmax_node);
    }

    for (uint32_t i = 0; i < images.get_number_classes() ; i++) {
        all_edges.push_back(new CNN_Edge(input_node, all_nodes[i + 1] /*ith softmax node*/, true, edge_innovation_count));
        total_weights += all_edges.back()->get_number_weights();
        edge_innovation_count++;
    }

    long genome_seed = rng_long(generator);
    cout << "seeding genome with: " << genome_seed << endl;

    CNN_Genome *first_genome = new CNN_Genome(genome_seed, epochs, all_nodes, all_edges);
    cout << "first genome fitness (before backprop): " << first_genome->get_fitness() << endl;

    //first_genome->write_to_file("input_cnn.txt");
    //first_genome->stochastic_backpropagation(images, "test_checkpoint.txt", "test_output.txt");

    genomes.push_back(first_genome);
}

CNN_Genome* EXACT::get_best_genome() {
    return genomes[0];
}

void EXACT::insert_genome(CNN_Genome* genome) {
    if (genome->get_fitness() < genomes[0]->get_fitness()) {
        cout << "NEW BEST FITNESS:" << genome->get_fitness() << endl;
        genome->write_to_file("global_best.txt");
    }

    cout << "inserted genome with best fitness found: " << genome->get_fitness() << endl;
    genomes.insert( upper_bound(genomes.begin(), genomes.end(), genome, sort_genomes_by_fitness()), genome);

    if (genomes.size() > population_size) {
        CNN_Genome *worst = genomes.back();
        cout << "deleting worst genome with fitness: " << worst->get_fitness() << endl;
        genomes.pop_back();

        delete worst;
    }
    
    cout << "genome fitnesses:" << endl;
    for (uint32_t i = 0; i < genomes.size(); i++) {
        cout << "\t" << i << " -- " << genomes[i]->get_fitness() << endl;
    }
    cout << endl;
}

CNN_Genome* EXACT::create_mutation() {
    //mutation options:
    //edges:
    //  1. disable edge (but make sure output node is still reachable)
    //  2. split edge
    //  3. add edge (make sure it does not exist already)
    //  4. increase/decrease stride (not yet)
    //nodes:
    //  1. increase/decrease size_x
    //  2. increase/decrease size_y
    //  3. increase/decrease max_pool (not yet)

    double edge_disable = 1.0;
    double edge_enable = 2.0;
    double edge_split = 3.0;
    double edge_add = 5.0;
    double edge_change_stride = 0.0;
    double node_change_size_x = 2.0;
    double node_change_size_y = 2.0;
    double node_change_pool_size = 0.0;

    double total = edge_disable + edge_enable + edge_split + edge_add + edge_change_stride +
                   node_change_size_x + node_change_size_y + node_change_pool_size;

    edge_disable /= total;
    edge_enable /= total;
    edge_split /= total;
    edge_add /= total;
    edge_change_stride /= total;
    node_change_size_x /= total;
    node_change_size_y /= total;
    node_change_pool_size /= total;

    /*
    cout << "mutating genome!" << endl;
    cout << "\tprobabilities: " << endl;
    cout << "\t\tedge_disable: " << edge_disable << endl;
    cout << "\t\tedge_split: " << edge_split << endl;
    cout << "\t\tedge_add: " << edge_add << endl;
    cout << "\t\tedge_change_stride: " << edge_change_stride << endl;
    cout << "\t\tnode_change_size_x: " << node_change_size_x << endl;
    cout << "\t\tnode_change_size_y: " << node_change_size_y << endl;
    cout << "\t\tnode_change_pool_size: " << node_change_pool_size << endl;
    */

    long child_seed = rng_long(generator);

    CNN_Genome *parent = genomes[rng_double(generator) * genomes.size()];

    CNN_Genome *child = new CNN_Genome(child_seed, epochs, parent->get_nodes(), parent->get_edges());

    if (genomes.size() == 1) return child;

    int number_mutations = 1;
    int modifications = 0;

    while (modifications < number_mutations) {
        double r = rng_double(generator);

        cerr << "\tr: " << r << endl;

        if (r < edge_disable) {
            cout << "\tDISABLING EDGE!" << endl;

            int edge_position = rng_double(generator) * child->get_number_edges();
            if (child->disable_edge(edge_position)) {
                modifications++;
            }

            continue;
        } 
        r -= edge_disable;

        if (r < edge_enable) {
            cout << "\tENABLING EDGE!" << endl;

            vector< CNN_Edge* > disabled_edges;

            for (uint32_t i = 0; i < child->get_number_edges(); i++) {
                CNN_Edge* current = child->get_edge(i);

                if (current == NULL) {
                    cout << "ERROR! edge " << i << " became null on child!" << endl;
                    exit(1);
                }

                if (current->is_disabled()) {
                    disabled_edges.push_back(current);
                }
            }
            
            if (disabled_edges.size() > 0) {
                int edge_position = rng_double(generator) * disabled_edges.size();

                cout << "\t\tenabling edge: " << disabled_edges[edge_position]->get_innovation_number() << endl;
                disabled_edges[edge_position]->enable();
                modifications++;
            } else {
                cout << "\t\tcould not enable an edge as there were no disabled edges!" << endl;
            }

            continue;
        } 
        r -= edge_enable;


        if (r < edge_split) {
            int edge_position = rng_double(generator) * child->get_number_edges();
            cout << "\tSPLITTING EDGE IN POSITION: " << edge_position << "!" << endl;

            CNN_Edge* edge = child->get_edge(edge_position);

            CNN_Node* input_node = edge->get_input_node();
            CNN_Node* output_node = edge->get_output_node();

            double depth = (input_node->get_depth() + output_node->get_depth()) / 2.0;
            int size_x = (input_node->get_size_x() + output_node->get_size_x()) / 2.0;
            int size_y = (input_node->get_size_y() + output_node->get_size_y()) / 2.0;

            CNN_Node *child_node = new CNN_Node(node_innovation_count, depth, size_x, size_y, HIDDEN_NODE);
            node_innovation_count++;

            //add two new edges, disable the split edge
            cout << "\t\tcreating edge " << edge_innovation_count << endl;
            CNN_Edge *edge1 = new CNN_Edge(input_node, child_node, false, edge_innovation_count);
            edge_innovation_count++;

            cout << "\t\tcreating edge " << edge_innovation_count << endl;
            CNN_Edge *edge2 = new CNN_Edge(child_node, output_node, false, edge_innovation_count);
            edge_innovation_count++;

            cout << "\t\tdisabling edge " << edge->get_innovation_number() << endl;
            edge->disable();

            child->add_node(child_node);
            child->add_edge(edge1);
            child->add_edge(edge2);

            //make sure copies are added to all_edges and all_nodes
            CNN_Node *node_copy = child_node->copy();
            CNN_Edge *edge_copy_1 = edge1->copy();
            CNN_Edge *edge_copy_2 = edge2->copy();

            //insert the new node into the population in sorted order
            all_nodes.insert( upper_bound(all_nodes.begin(), all_nodes.end(), node_copy, sort_CNN_Nodes_by_depth()), node_copy);
            edge_copy_1->set_nodes(all_nodes);
            edge_copy_2->set_nodes(all_nodes);

            all_edges.insert( upper_bound(all_edges.begin(), all_edges.end(), edge_copy_1, sort_CNN_Edges_by_depth()), edge_copy_1);
            all_edges.insert( upper_bound(all_edges.begin(), all_edges.end(), edge_copy_2, sort_CNN_Edges_by_depth()), edge_copy_2);

            modifications++;

            continue;
        }
        r -= edge_split;

        if (r < edge_add) {
            cout << "\tADDING EDGE!" << endl;

            CNN_Node *node1;
            CNN_Node *node2;

            do {
                int r1 = rng_double(generator) * all_nodes.size();
                int r2 = rng_double(generator) * all_nodes.size() - 1;

                if (r1 == r2) r2++;

                if (r1 > r2) {  //swap r1 and r2 so node2 is always deeper than node1
                    int temp = r1;
                    r1 = r2;
                    r2 = temp;
                }

                node1 = all_nodes[r1];
                node2 = all_nodes[r2];
            } while (node1->get_depth() >= node2->get_depth());
            //after this while loop, node 2 will always be deeper than node 1

            int node1_innovation_number = node1->get_innovation_number();
            int node2_innovation_number = node2->get_innovation_number();

            //check to see if the edge already exists
            bool edge_exists = false;
            for (uint32_t i = 0; i < all_edges.size(); i++) {
                if (all_edges[i]->connects(node1_innovation_number, node2_innovation_number)) {
                    edge_exists = true;
                    break;
                }
            }

            //TODO: make sure that both nodes exist in the child!!!
            if (!edge_exists) {
                cout << "\t\tadding edge between node innovation numbers " << node1_innovation_number << " and " << node2_innovation_number << endl;

                CNN_Edge *edge = new CNN_Edge(node1, node2, false, edge_innovation_count);
                //insert edge in order of depth
                all_edges.insert( upper_bound(all_edges.begin(), all_edges.end(), edge, sort_CNN_Edges_by_depth()), edge);
                edge_innovation_count++;

                CNN_Edge *child_edge = edge->copy();
                if (!child_edge->set_nodes(child->get_nodes())) {
                    child_edge->reinitialize(generator);
                }

                child->add_edge(child_edge);

                modifications++;
            } else {
                cout << "\t\tnot adding edge between node innovation numbers " << node1_innovation_number << " and " << node2_innovation_number << " because edge already exists!" << endl;
            }

            continue;
        }
        r -= edge_add;

        if (r < edge_change_stride) {
            cout << "\tCHANGING EDGE STRIDE!" << endl;

            //child->mutate(MUTATE_EDGE_STRIDE, node_innovation_count, edge_innovation_count);

            continue;
        }
        r -= edge_change_stride;

        if (r < node_change_size_x) {
            cout << "\tCHANGING NODE SIZE X!" << endl;

            if (child->get_number_softmax_nodes() + 1 == child->get_number_nodes()) {
                cout << "\t\tno non-input or softmax nodes so cannot change node size" << endl;
                continue;
            }

            //should have a value between -2 and 2 (inclusive)
            int change = (2 * rng_double(generator)) + 1;
            if (rng_double(generator) < 0.5) change *= -1;

            //make sure we don't change the size of the input node
            int r = (rng_double(generator) * (child->get_number_nodes() - 1 - child->get_number_softmax_nodes())) + 1;

            CNN_Node *modified_node = child->get_node(r);

            if (modified_node->modify_size_x(change, generator)) {
                //need to make sure all edges with this as it's input or output get updated
                child->resize_edges_around_node( modified_node->get_innovation_number() );
                modifications++;

                cout << "\t\tmodified size x by " << change << endl;
            } else {
                cout << "\t\tmodification resulted in no change" << endl;
            }

            continue;
        }
        r -= node_change_size_x;

        if (r < node_change_size_y) {
            cout << "\tCHANGING NODE SIZE Y!" << endl;

            if (child->get_number_softmax_nodes() + 1 == child->get_number_nodes()) {
                cout << "\t\tno non-input or softmax nodes so cannot change node size" << endl;
                continue;
            }

            //should have a value between -2 and 2 (inclusive)
            int change = (2 * rng_double(generator)) + 1;
            if (rng_double(generator) < 0.5) change *= -1;

            //make sure we don't change the size of the input node
            int r = (rng_double(generator) * (child->get_number_nodes() - 1 - child->get_number_softmax_nodes())) + 1;

            CNN_Node *modified_node = child->get_node(r);

            if (modified_node->modify_size_y(change, generator)) {
                //need to make sure all edges with this as it's input or output get updated
                child->resize_edges_around_node( modified_node->get_innovation_number() );
                modifications++;

                cout << "\t\tmodified size y by " << change << endl;
            } else {
                cout << "\t\tmodification resulted in no change" << endl;
            }

            continue;
        }
        r -= node_change_size_y;

        if (r < node_change_pool_size) {
            cout << "\tCHANGING NODE POOL SIZE!" << endl;

            //child->mutate(MUTATE_NODE_POOL_SIZE, node_innovation_count, edge_innovation_count);

            continue;
        }
        r -= node_change_pool_size;

        cerr << "ERROR: problem choosing mutation type -- should never get here!" << endl;
        cerr << "\tremaining random value (for mutation selection): " << r << endl;
        exit(1);
    }

    if (!child->sanity_check()) {
        cerr << "ERROR: child failed sanity check!" << endl;
        exit(1);
    }

    if (!child->outputs_connected()) {
        cerr << "\tAll softmax nodes were not reachable, deleting child." << endl;
        delete child;
        return NULL;
    }

    if (genomes.size() < population_size) {
        //insert a copy with a bad fitness so we have more things to generate new genomes with
        insert_genome(new CNN_Genome(child_seed, epochs, child->get_nodes(), child->get_edges()));
     }

    return child;
}

CNN_Genome* EXACT::create_child() {
    return NULL;
}