package ged.structures.graph;

import ged.alg.distance.GraphDistance;

public class LabeledVertex extends Vertex
{
  private final String label;
  public LabeledVertex(String label)
  {
    this.label = label;
  }

  @Override
  public String toString()
  {
    return label;
  }

  @Override
  public double getDistance(Vertex v)
  {

    return v.toString().equals(this.toString()) ? 0
        : GraphDistance.VERTEX_SUBSTITUTION_COSTS;
  }

  @Override
  public double getDeletionCost()
  {
    return GraphDistance.VERTEX_INSERTION_DELETION_COSTS;
  }
}
