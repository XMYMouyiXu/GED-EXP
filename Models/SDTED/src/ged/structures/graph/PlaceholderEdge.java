package ged.structures.graph;

import ged.alg.distance.GraphDistance;

public class PlaceholderEdge extends Edge
{
  private static final String PLACEHOLDER_STRING = "";

  public PlaceholderEdge()
  {}

  @Override
  public String toString()
  {
    return PLACEHOLDER_STRING;
  }

  @Override
  public double getDistance(Edge v)
  {
    return 0;
  }

  @Override
  public double getDeletionCost()
  {
    return GraphDistance.EDGE_INSERTION_DELETION_COSTS;
  }

  @Override
  public Edge copy()
  {
    return new PlaceholderEdge();
  }

}
