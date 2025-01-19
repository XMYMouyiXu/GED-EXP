package ged.alg.distance;

import java.util.List;

import ged.alg.distance.tree.SDTEDCached;
import ged.alg.distance.tree.SDTEDUncached;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;
import ged.structures.tree.CTree;

public abstract class SDTEDbasedDistance extends GraphDistance
{
  protected final double layerWeightFactor;
  private boolean useCache = true;

  protected SDTEDbasedDistance(double layerWeightFactor)
  {
    this.layerWeightFactor = layerWeightFactor;
  }

  public void setCachedAssignment(boolean useCache)
  {
    this.useCache = useCache;
  }

  protected <V extends Vertex, E extends Edge> int[] assignment(
      List<CTree<V, E>> trees1, List<CTree<V, E>> trees2,
      double layerWeightFactor)
  {
    return useCache ? SDTEDCached.assignment(trees1, trees2, layerWeightFactor)
        : SDTEDUncached.assignment(trees1, trees2, layerWeightFactor);
  }
}
