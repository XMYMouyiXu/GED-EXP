package ged.alg.distance;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.jgrapht.graph.AbstractGraph;

import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;

public abstract class GraphDistance {

  public static final double VERTEX_INSERTION_DELETION_COSTS = 1;
  public static final double VERTEX_SUBSTITUTION_COSTS = 1;
  public static final double EDGE_INSERTION_DELETION_COSTS = 1;
  public static final double EDGE_SUBSTITUTION_COSTS = 1;

  public abstract <V extends Vertex, E extends Edge> double computeGraphDistance(AbstractGraph<V, E> g1, AbstractGraph<V, E> g2);

  /**
   * @param g1
   * @param g2
   * @param vertices1
   *          vertex list for g1, for vertex ordering
   * @param vertices1
   *          vertex list for g2, for vertex ordering
   * @param c
   *          the cost matrix between vertices of g1 and vertices of g2
   * @param assignment
   *          mapping between the vertices of the two graphs (1 to g2)
   * @return the edit costs derived from the assignment
   */
  public static <V extends Vertex, E extends Edge> double computeCostOfEditPath(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2, List<V> vertices1,
      List<V> vertices2, int[] assignment)
  {
    int v1Size = vertices1.size();
    int v2Size = vertices2.size();

    Map<V, Integer> v1Map = new HashMap<V, Integer>(v1Size);
    for (int i = 0; i < v1Size; ++i)
      v1Map.put(vertices1.get(i), i);

    double r = 0;

    HashSet<V> notDeleted = new HashSet<V>();
    // vertex deletions/relabeling
    for (V v1 : vertices1)
    {
      int v2Index = assignment[v1Map.get(v1)];
      if (v2Index >= v2Size)
      {
        // vertex deleted
        r += VERTEX_INSERTION_DELETION_COSTS;
      }
      else
      {
        // vertex relabeled if necessary
        V v2 = vertices2.get(v2Index);
        r += v1.getDistance(v2);
        notDeleted.add(v2);
      }
    }

    // vertex insertions
    r += VERTEX_INSERTION_DELETION_COSTS
        * (v2Size - notDeleted.size());

    // edge relabeling and deletion
    HashSet<E> notDeletedEdges = new HashSet<E>();

    for (E e1 : g1.edgeSet())
    {
      int u2Index = assignment[v1Map.get(g1.getEdgeSource(e1))];
      int v2Index = assignment[v1Map.get(g1.getEdgeTarget(e1))];
      E e2 = null;
      if (u2Index < v2Size && v2Index < v2Size)
      {
        V u2 = vertices2.get(u2Index);
        V v2 = vertices2.get(v2Index);
        e2 = g2.getEdge(u2, v2);
      }
      if (e2 != null)
      {
        // both vertices are not deleted and edge between the assigned vertices
        // exists
        r += e1.getDistance(e2);
        notDeletedEdges.add(e2);
      }
      else
      {
        // edge has to be deleted
        r += EDGE_INSERTION_DELETION_COSTS;
      }
    }

    // edge insertion
    r += EDGE_INSERTION_DELETION_COSTS
        * (g2.edgeSet().size() - notDeletedEdges.size());

    return r;
  }
  public abstract String getID();
}
