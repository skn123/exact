#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <vector>
using std::vector;


#include "node_pheromone.hxx"


NODE_Pheromones::NODE_Pheromones(double *_type_pheromones, vector<EDGE_Pheromone*> *_pheromone_lines , int _layer_type, int _current_layer) :type_pheromones(_type_pheromones), pheromone_lines(_pheromone_lines), layer_type(_layer_type), current_layer(_current_layer){

  type_pheromones = _type_pheromones;
  pheromone_lines = _pheromone_lines;
  layer_type      = _layer_type;
  current_layer   = _current_layer;
}

int32_t NODE_Pheromones::get_layer_type(){
  return layer_type;
}

int32_t NODE_Pheromones::get_current_layer(){
  return current_layer;
}

vector<EDGE_Pheromone*>* NODE_Pheromones::get_pheromone_lines(){
  return pheromone_lines;
}

NODE_Pheromones::~NODE_Pheromones(){
  // delete type_pheromones;
}

// NODE_Pheromones::double* get_pheromones(){
//   double p[pheromone_lines.size()]
//   for (int i=0, i<pheromone_lines().size, i++){
//     p[i] = pheromone_lines[i].get_edge_phermone();
//   }
//   return p
// }
