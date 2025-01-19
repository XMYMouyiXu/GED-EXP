package ged.alg.distance.comparison;

import java.util.List;

import org.jgrapht.graph.AbstractGraph;

import com.google.common.collect.Lists;

import ged.alg.assignment.HungarianAlgorithm;
import ged.alg.distance.GraphDistance;
import ged.alg.distance.comparison.walks.WalkHistograms;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;

public class WalksMatching extends GraphDistance
{
  @Override
  public String getID()
  {
    return "WalksMatching_"+k;
  }
  private int k;
  public WalksMatching(int k)
  {
	  this.k = k;
  }

  @Override
  public <V extends Vertex, E extends Edge> double computeGraphDistance(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    // lists for vertex ordering
    List<V> vertices1 = Lists.newArrayList(g1.vertexSet());
    List<V> vertices2 = Lists.newArrayList(g2.vertexSet());

    double[][] costMatrix = computeCostMatrix(g1, vertices1, g2, vertices2,
        this.k);

    return computeCostOfEditPath(g1, g2, vertices1, vertices2,
        assignment(costMatrix));
  }


  private <V extends Vertex, E extends Edge> double[][] computeCostMatrix(
      AbstractGraph<V, E> g1, List<V> vertices1, AbstractGraph<V, E> g2,
      List<V> vertices2, int k)
  {
    int n = g1.vertexSet().size();
    int m = g2.vertexSet().size();

    double[][] C = new double[n + m][n + m];

    WalkHistograms<V,E> h = new WalkHistograms<V,E>(this.k,g1, vertices1, g2, vertices2);
    h.init();
    // vertex matching (upper left corner)
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < m; j++)
      {
        V v1 = vertices1.get(i);
        V v2 = vertices2.get(j);
        C[i][j] = h.vertexSubstitution(v1, v2);
      }
    }

    // vertex insertion (lower left corner)
    for (int i = n; i < n + m; i++)
    {
      for (int j = 0; j < m; j++)
      {
        C[i][j] = Double.POSITIVE_INFINITY;
      }
    }
    for (int j = 0; j < m; j++)
    {
      V v2 = vertices2.get(j);
      C[n + j][j] = h.vertexInsertionDeletion(v2);
    }

    // vertex deletion (upper right corner)
    for (int i = 0; i < n; i++)
    {
      for (int j = m; j < n + m; j++)
      {
        C[i][j] = Double.POSITIVE_INFINITY;
      }
    }
    for (int i = 0; i < n; i++)
    {
      V v1 = vertices1.get(i);
      C[i][m + i] = h.vertexInsertionDeletion(v1);
    }

    // zero fill-in (lower right corner)
    for (int i = n; i < n + m; i++)
    {
      for (int j = m; j < n + m; j++)
      {
        C[i][j] = 0;
      }
    }

   // MathUtil.printMatrix(C);
    return C;
  }

  private static final int[] assignment(double[][] C)
  {
    return HungarianAlgorithm.assignment(C);
  }

}
