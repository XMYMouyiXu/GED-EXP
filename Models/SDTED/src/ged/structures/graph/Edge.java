package ged.structures.graph;

public abstract class Edge
{
  /**
   * this distance must be symmetric
   */
  public abstract double getDistance(Edge v);

  public abstract double getDeletionCost();

  public abstract Edge copy();

}
