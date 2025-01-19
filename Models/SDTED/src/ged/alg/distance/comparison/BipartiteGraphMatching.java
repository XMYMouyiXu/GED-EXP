package ged.alg.distance.comparison;

import java.util.List;

import org.jgrapht.graph.AbstractGraph;

import com.google.common.collect.Lists;

import ged.alg.assignment.HungarianAlgorithm;
import ged.alg.distance.GraphDistance;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;

public class BipartiteGraphMatching extends GraphDistance
{
  @Override
  public String getID()
  {
    return "BipartiteGraphMatching";
  }

  @Override
  public <V extends Vertex, E extends Edge> double computeGraphDistance(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    // lists for vertex ordering
    List<V> vertices1 = Lists.newArrayList(g1.vertexSet());
    List<V> vertices2 = Lists.newArrayList(g2.vertexSet());

    double[][] costMatrix = computeCostMatrix(g1, vertices1, g2, vertices2,
        false);

    return computeCostOfEditPath(g1, g2, vertices1, vertices2,
        assignment(costMatrix));
  }

  public static <V extends Vertex, E extends Edge> double computeBranchLowerBound(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    // lists for vertex ordering
    List<V> vertices1 = Lists.newArrayList(g1.vertexSet());
    List<V> vertices2 = Lists.newArrayList(g2.vertexSet());

    double[][] costMatrix = computeCostMatrix(g1, vertices1, g2, vertices2,
        true);

    return assignmentCost(costMatrix);
  }

  private static <V extends Vertex, E extends Edge> double[][] computeCostMatrix(
      AbstractGraph<V, E> g1, List<V> vertices1, AbstractGraph<V, E> g2,
      List<V> vertices2, boolean branch)
  {
    int n = g1.vertexSet().size();
    int m = g2.vertexSet().size();

    double[][] C = new double[n + m][n + m];

    // vertex matching (upper left corner)
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < m; j++)
      {
        V v1 = vertices1.get(i);
        V v2 = vertices2.get(j);
        C[i][j] = vertexSubstitution(v1, g1, v2, g2, branch);
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
      C[n + j][j] = vertexInsertionDeletion(v2, g2, branch);
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
      C[i][m + i] = vertexInsertionDeletion(v1, g1, branch);
    }

    // zero fill-in (lower right corner)
    for (int i = n; i < n + m; i++)
    {
      for (int j = m; j < n + m; j++)
      {
        C[i][j] = 0;
      }
    }

    return C;
  }

  private static <V extends Vertex, E extends Edge> double vertexInsertionDeletion(
      V v, AbstractGraph<V, E> g, boolean branch)
  {
    double rV = v.getDeletionCost();
    double rE = 0;
    // add costs for all edge deletions
    for (E e : g.edgesOf(v))
    {
      rE += e.getDeletionCost();
    }
    if (branch)
    {
      rE = 0.5 * rE;
    }
    return rV + rE;
  }

  private static <V extends Vertex, E extends Edge> double vertexSubstitution(
      V v1, AbstractGraph<V, E> g1, V v2, AbstractGraph<V, E> g2,
      boolean branch)
  {
    double cV = v1.getDistance(v2);

    // add lower bound for edge costs obtained from assignment (might be not
    // necessary if we dont have special edge costs
    int n = g1.degreeOf(v1);
    int m = g2.degreeOf(v2);

    // no edges
    if (n == 0 && m == 0) return cV;

    double[][] C = new double[n + m][n + m];

    // upper left corner
    int e1Id = 0;
    for (E e1 : g1.edgesOf(v1))
    {
      int e2Id = 0;
      for (E e2 : g2.edgesOf(v2))
      {
        C[e1Id][e2Id] = e1.getDistance(e2);
        e2Id++;
      }
      e1Id++;
    }

    // upper right corner
    for (int i = 0; i < n; i++)
    {
      for (int j = m; j < n + m; j++)
      {
        C[i][j] = Double.POSITIVE_INFINITY;
      }
    }
    e1Id = 0;
    for (E e1 : g1.edgesOf(v1))
    {
      C[e1Id][m + e1Id] = e1.getDeletionCost();
      e1Id++;
    }

    // lower left corner
    for (int i = n; i < n + m; i++)
    {
      for (int j = 0; j < m; j++)
      {
        C[i][j] = Double.POSITIVE_INFINITY;
      }
    }
    int e2Id = 0;
    for (E e2 : g2.edgesOf(v2))
    {
      C[n + e2Id][e2Id] = e2.getDeletionCost();
      e2Id++;
    }

    // lower right corner
    for (int i = n; i < n + m; i++)
    {
      for (int j = m; j < n + m; j++)
      {
        C[i][j] = 0;
      }
    }

    double cE = assignmentCost(C);

    if (branch) // branch lower bound works with halved edge costs
    {
      cE = 0.5 * cE;
    }
    return cV + cE;
  }

  private static final double assignmentCost(double[][] C)
  {
    return HungarianAlgorithm.assignmentCost(C);
  }

  private static final int[] assignment(double[][] C)
  {
    return HungarianAlgorithm.assignment(C);
  }

}
