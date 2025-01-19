package ged.alg.distance;

import java.util.List;

import org.jgrapht.graph.AbstractGraph;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import ged.alg.graphencoder.tree.NeighborhoodTreeV2;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;
import ged.structures.graph.VertexWrapper;
import ged.structures.tree.CTree;

public class SDTEDbasedDistanceV2 extends SDTEDbasedDistance
{
  private final int iterations;

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
        new NeighborhoodTreeV2<V, E>(g1, iterations)//
        .getCurrentTreeRepresentations();

    List<CTree<VertexWrapper<V>, Edge>> trees2 = //
        new NeighborhoodTreeV2<V, E>(g2, iterations)//
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

  public SDTEDbasedDistanceV2(int i)
  {
    super(1);
    this.iterations = i;
  }

  public SDTEDbasedDistanceV2()
  {
    super(1);
    this.iterations = -1;
  }

  public SDTEDbasedDistanceV2(int i, double layerWeightFactor)
  {
    super(layerWeightFactor);
    this.iterations = i;
  }

  @Override
  public String getID()
  {
    return "SDTEDbasedDistanceV2_" + this.iterations + "_"
        + this.layerWeightFactor;
  }

}
