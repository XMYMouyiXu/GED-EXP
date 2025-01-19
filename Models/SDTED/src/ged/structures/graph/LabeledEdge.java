package ged.structures.graph;

import ged.alg.distance.GraphDistance;

public class LabeledEdge extends Edge
{
  public static final String DEFAULT_UNSET_LABEL = "?";

  private final String label;

  public LabeledEdge()
  {
    this.label = DEFAULT_UNSET_LABEL;
  }

  public LabeledEdge(String label)
  {
    this.label = label;
  }

  @Override
  public String toString()
  {
    return label;
  }

  @Override
  public double getDistance(Edge v)
  {
    return toString().equals(v.toString()) ? 0
        : GraphDistance.EDGE_SUBSTITUTION_COSTS;
  }

  @Override
  public double getDeletionCost()
  {
    return GraphDistance.EDGE_INSERTION_DELETION_COSTS;
  }

  @Override
  public LabeledEdge copy()
  {
    return new LabeledEdge(label);
  }
}
