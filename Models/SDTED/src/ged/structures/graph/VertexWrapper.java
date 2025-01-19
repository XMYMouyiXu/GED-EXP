package ged.structures.graph;

import ged.alg.distance.GraphDistance;

public class VertexWrapper<V> extends Vertex
{
  public V v;
  public VertexWrapper(V wrappedObject)
  {
    v = wrappedObject;
  }

  public V getObject()
  {
    return v;
  }

  @Override
  public String toString()
  {
    return v.toString();
  }

  @Override
  public double getDistance(Vertex other)
  {
    return toString().equals(other.toString()) ? 0
        : GraphDistance.VERTEX_SUBSTITUTION_COSTS;
  }

  @Override
  public double getDeletionCost()
  {
    return GraphDistance.VERTEX_INSERTION_DELETION_COSTS;
  }
}
