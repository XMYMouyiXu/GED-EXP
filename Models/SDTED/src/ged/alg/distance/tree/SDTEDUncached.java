package ged.alg.distance.tree;

import java.util.List;

import ged.alg.assignment.HungarianAlgorithm;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;
import ged.structures.tree.CTree;
import ged.structures.tree.Tree;
import ged.structures.tree.Tree.ChildWithEdge;

public class SDTEDUncached
{
  private final double layerWeightFactor;

  public SDTEDUncached()
  {
    layerWeightFactor = 1;
  }

  public SDTEDUncached(double layerWeightFactor)
  {
    this.layerWeightFactor = layerWeightFactor;
  }

  // Structure and depth preserving tree edit distance as described in
  // A Generalized Weisfeiler-Lehman Graph Kernel
  // https://arxiv.org/abs/2101.08104

  /**
   * @param t1
   * @param root1
   * @param t2
   * @param root2
   * @return computes the structure and depth preserving tree edit distance
   *         between subtree of t1 rooted at root and subtree of t2 rooted at
   *         root2
   */
  private <V extends Vertex, E extends Edge> double computeSDTED(Tree<V, E> t1,
      V root1, Tree<V, E> t2, V root2)
  {
    final double cost;
    if (root1 == null)//
      cost = root2 == null ? 0 : deletion(t2, root2);// symmetry
    else if (root2 == null)//
      cost = deletion(t1, root1);// symmetry
    else
      cost = root1.getDistance(root2)// distance between roots
      + computeSDM(t1, root1, t2, root2) * layerWeightFactor;

    return cost;
  }

  /**
   * @param t
   * @param v
   * @return returns the costs for deleting the subtree of t rooted at v
   */
  private <V extends Vertex, E extends Edge> double deletion(Tree<V, E> t, V v)
  {
    double childCost = 0;
    for (ChildWithEdge<V, E> child : t.getChildrenWithEdges(v))
      childCost += deletion(t, child.v) + child.e.getDeletionCost();

    return v.getDeletionCost() + childCost * layerWeightFactor;
  }

  /**
   * @param t1
   * @param root1
   * @param t2
   * @param root2
   * @return computes the cost of an optimal assignment between the children of
   *         root1 and root2
   */
  private <V extends Vertex, E extends Edge> double computeSDM(Tree<V, E> t1,
      V root1, Tree<V, E> t2, V root2)
  {
    List<ChildWithEdge<V, E>> children1 = t1.getChildrenWithEdges(root1);
    List<ChildWithEdge<V, E>> children2 = t2.getChildrenWithEdges(root2);

    int n = children1.size();
    int m = children2.size();
    int max = Math.max(n, m);
    // if both vertices have no children we can return 0
    if (max == 0) return 0;

    // compute costmatrix
    double[][] c = new double[max][max];
    for (int i = 0; i < n; ++i)
    {
      ChildWithEdge<V, E> child1 = children1.get(i);
      for (int j = 0; j < m; ++j)
      {
        ChildWithEdge<V, E> child2 = children2.get(j);
        c[i][j] = computeSDTED(t1, child1.v, t2, child2.v)
            + child1.e.getDistance(child2.e);
      }
    }

    if (n < m)
    {
      for (int j = 0; j < max; ++j)
      {
        ChildWithEdge<V, E> child = children2.get(j);
        double cost = computeSDTED(null, null, t2, child.v)
            + child.e.getDeletionCost(); // symmetry
        for (int i = n; i < max; ++i)
          c[i][j] = cost;
      }
    }
    else if (m < n)
    {
      for (int i = 0; i < max; ++i)
      {
        ChildWithEdge<V, E> child = children1.get(i);
        double cost = computeSDTED(t1, child.v, null, null)
            + child.e.getDeletionCost();// symmetry
        for (int j = m; j < max; ++j)
          c[i][j] = cost;
      }
    }

    // solve assignment and return costs
    return assignmentCost(c);
  }

  public <V extends Vertex, E extends Edge> double[][] computeCostMatrix(
      List<? extends Tree<V, E>> trees1, List<? extends Tree<V, E>> trees2)
  {
    // copy the lists so the outside reference is not modified
    int n = trees1.size();
    int m = trees2.size();

    int max = Math.max(n, m);

    // compute costmatrix
    double[][] c = new double[max][max];
    for (int i = 0; i < n; ++i)
    {
      Tree<V, E> t1 = trees1.get(i);
      V r1 = t1.getRoot();
      for (int j = 0; j < m; ++j)
      {
        Tree<V, E> t2 = trees2.get(j);
        c[i][j] = computeSDTED(t1, r1, t2, t2.getRoot());
      }
    }

    if (n < m)
    {
      for (int j = 0; j < max; ++j)
      {
        Tree<V, E> t2 = trees2.get(j);
        double cost = computeSDTED(null, null, t2, t2.getRoot());
        for (int i = n; i < max; ++i)
          c[i][j] = cost;
      }
    }
    else if (m < n)
    {
      for (int i = 0; i < max; ++i)
      {
        Tree<V, E> t1 = trees1.get(i);
        double cost = computeSDTED(t1, t1.getRoot(), null, null);
        for (int j = m; j < max; ++j)
          c[i][j] = cost;
      }
    }
    return c;
  }

  public static <V extends Vertex, E extends Edge> double[][] costMatrix(
      List<? extends Tree<V, E>> trees1, List<? extends Tree<V, E>> trees2)
  {
    return new SDTEDUncached().computeCostMatrix(trees1, trees2);
  }

  public static <V extends Vertex, E extends Edge> double[][] costMatrix(
      List<? extends Tree<V, E>> trees1, List<? extends Tree<V, E>> trees2,
          double layerWeightFactor)
  {
    return new SDTEDUncached(layerWeightFactor).computeCostMatrix(trees1,
        trees2);
  }

  public static <V extends Vertex, E extends Edge> int[] assignment(
      List<CTree<V, E>> trees1, List<CTree<V, E>> trees2)
  {
    return assignment(costMatrix(trees1, trees2));
  }

  public static <V extends Vertex, E extends Edge> int[] assignment(
      List<CTree<V, E>> trees1, List<CTree<V, E>> trees2,
      double layerWeightFactor)
  {
    return assignment(costMatrix(trees1, trees2, layerWeightFactor));
  }

  private static double assignmentCost(double[][] C)
  {
    // VolgenantJonker vj = new VolgenantJonker(C);
    // return vj.computeAssignment();
    return HungarianAlgorithm.assignmentCost(C);
  }

  private static int[] assignment(double[][] C)
  {
    // VolgenantJonker vj = new VolgenantJonker(C);
    // vj.computeAssignment();
    // return vj.getAssignment();
    return HungarianAlgorithm.assignment(C);
  }
}