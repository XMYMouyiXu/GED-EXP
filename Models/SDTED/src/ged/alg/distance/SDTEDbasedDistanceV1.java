package ged.alg.distance;

import java.util.List;

import org.jgrapht.graph.AbstractGraph;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import ged.alg.graphencoder.tree.NeighborhoodTreeV1;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;
import ged.structures.tree.CTree;

public class SDTEDbasedDistanceV1 extends SDTEDbasedDistance
{
  private final int iterations;

  public SDTEDbasedDistanceV1(int i)
  {
    super(1);
    this.iterations = i;
  }

  public SDTEDbasedDistanceV1(int i, double layerWeightFactor)
  {
    super(layerWeightFactor);
    this.iterations = i;
  }

  public SDTEDbasedDistanceV1()
  {
    super(1);
    this.iterations = -1;
  }

  /**
   * @param g1
   * @param g2
   * @return Computes a distance between g1 and g2.
   */
  @Override
  public <V extends Vertex, E extends Edge> double computeGraphDistance(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {

    List<CTree<V, Edge>> trees1 = //
        new NeighborhoodTreeV1<V, E>(g1, iterations)//
        .getCurrentTreeRepresentations();

    List<CTree<V, Edge>> trees2 = //
        new NeighborhoodTreeV1<V, E>(g2, iterations)//
        .getCurrentTreeRepresentations();

    // lists for vertex ordering
    List<V> vertices1 = Lists
        .newArrayList(Iterables.transform(trees1, CTree::getRoot));
    List<V> vertices2 = Lists
        .newArrayList(Iterables.transform(trees2, CTree::getRoot));

    // solve assignment and return cost of edit path
    return computeCostOfEditPath(g1, g2, vertices1, vertices2,
        assignment(trees1, trees2, this.layerWeightFactor));

  }

  @Override
  public String getID()
  {
    return "SDTEDbasedDistanceV1_" + this.iterations + "_"
        + this.layerWeightFactor;
  }

}
