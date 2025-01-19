package ged.alg.distance.comparison;


import org.jgrapht.graph.AbstractGraph;

import ged.alg.distance.GraphDistance;
import ged.alg.distance.comparison.beamsearch.BeamSearch;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;

public class BeamSearchDistance<V extends Vertex, E extends Edge> extends GraphDistance
{
  @Override
  public String getID()
  {
    return "BeamSearch_"+s;
  }
  private int s;
  private BeamSearch search;
  public BeamSearchDistance(int s)
  {
	  this.s = s;
	  this.search = new BeamSearch<V,E>(s);
  }

  @Override
  public <V extends Vertex, E extends Edge> double computeGraphDistance(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    return this.search.getEditDistance(g1,g2, g1.vertexSet().size()+g2.vertexSet().size()+g1.edgeSet().size()+g2.edgeSet().size());
  }
}
