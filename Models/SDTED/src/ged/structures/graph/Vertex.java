package ged.structures.graph;

public abstract class Vertex {

  /**
   * this distance must be symmetric
   */
  public abstract double getDistance(Vertex v);

  public abstract double getDeletionCost();

}
