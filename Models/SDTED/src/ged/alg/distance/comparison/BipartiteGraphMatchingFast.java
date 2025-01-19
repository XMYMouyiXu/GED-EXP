package ged.alg.distance.comparison;

import java.util.List;

import org.jgrapht.graph.AbstractGraph;

import com.google.common.collect.Lists;

import ged.alg.assignment.HungarianAlgorithm;
import ged.alg.distance.GraphDistance;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;

public class BipartiteGraphMatchingFast extends GraphDistance
{

  @Override
  public String getID()
  {
    return "BipartiteGraphMatchingFast";
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
    int max = Math.max(n, m);

    double[][] C = new double[max][max];

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

    for (int i = n; i < max; ++i)
    {
      for (int j = 0; j < max; ++j)
      {
        V v = vertices2.get(j);
        C[i][j] = vertexInsertionDeletion(v, g2, branch);
      }
    }

    for (int j = m; j < max; ++j)
    {
      for (int i = 0; i < max; ++i)
      {
        V v = vertices1.get(i);
        C[i][j] = vertexInsertionDeletion(v, g1, branch);
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
    int max = Math.max(n, m);

    // no edges
    if (n == 0 && m == 0) return cV;

    double[][] C = new double[max][max];

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

    e1Id = n;
    for (int i = n; i < max; ++i)
    {
      int e2Id = 0;
      for (E e2 : g2.edgesOf(v2))
      {
        C[e1Id][e2Id] = e2.getDeletionCost();
        e2Id++;
      }
      e1Id++;
    }

    int e2Id = m;
    for (int j = m; j < max; ++j)
    {
      e1Id = 0;
      for (E e1 : g1.edgesOf(v1))
      {
        C[e1Id][e2Id] = e1.getDeletionCost();
        e1Id++;
      }
      e2Id++;
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
    // VolgenantJonker vj = new VolgenantJonker(C);
    // return vj.computeAssignment();
    return HungarianAlgorithm.assignmentCost(C);
  }

  public static final int[] assignment(double[][] C)
  {
    // VolgenantJonker vj = new VolgenantJonker(C);
    // vj.computeAssignment();
    // return vj.getAssignment();
    return HungarianAlgorithm.assignment(C);
  }

}
