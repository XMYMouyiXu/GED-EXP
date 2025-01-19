package ged.alg.distance;

import java.util.List;

import org.jgrapht.graph.AbstractGraph;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import ged.alg.graphencoder.tree.NeighborhoodTreeWL;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;
import ged.structures.graph.VertexWrapper;
import ged.structures.tree.CTree;

public class SDTEDbasedDistanceWL extends SDTEDbasedDistance
{
  private final int iterations;

  public SDTEDbasedDistanceWL(int iterations)
  {
    super(1);
    this.iterations = iterations;
  }

  public SDTEDbasedDistanceWL(int iterations, double layerWeightFactor)
  {
    super(layerWeightFactor);
    this.iterations = iterations;
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
    List<CTree<VertexWrapper<V>, Edge>> trees1 = //
        new NeighborhoodTreeWL<V, E>(g1, iterations)//
        .getCurrentTreeRepresentations();

    List<CTree<VertexWrapper<V>, Edge>> trees2 = //
        new NeighborhoodTreeWL<V, E>(g2, iterations)//
        .getCurrentTreeRepresentations();

    // lists for vertex ordering
    List<V> vertices1 = Lists.newArrayList(
        Iterables.transform(trees1, g -> g.getRoot().getObject()));
    List<V> vertices2 = Lists.newArrayList(
        Iterables.transform(trees2, g -> g.getRoot().getObject()));

    // solve assignment and return cost of edit path
    return computeCostOfEditPath(g1, g2, vertices1, vertices2,
        assignment(trees1, trees2, this.layerWeightFactor));
  }

  @Override
  public String getID()
  {
    return "SDTEDbasedDistanceWL_" + this.iterations + "_"
        + this.layerWeightFactor;
  }
}
