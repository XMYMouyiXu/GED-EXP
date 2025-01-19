package ged.alg.distance.comparison;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

import org.jgrapht.alg.util.Pair;
import org.jgrapht.graph.AbstractGraph;

import ac.bss_ged.BSEditDistanceJNI;
import ged.alg.distance.GraphDistance;
import ged.structures.graph.Edge;
import ged.structures.graph.LabeledEdge;
import ged.structures.graph.LabeledVertex;
import ged.structures.graph.Vertex;

public class BSSExactGraphEditDistance extends GraphDistance
{
  @Override
  public String getID()
  {
    return "BSSEXACTGED";
  }
  int width;

  /**
   * @param width
   *          of the beam search
   */
  public BSSExactGraphEditDistance(int width)
  {
    this.width = width;
  }

  @Override
  public <V extends Vertex, E extends Edge> double computeGraphDistance(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    return compute(g1, g2, Double.MAX_VALUE);
  }

  public <V extends Vertex, E extends Edge> double compute(
      AbstractGraph<V, E> q, AbstractGraph<V, E> g, double bound)
  {

    // floor is allowed due to the uniform cost-model
    int intBound = (int) bound;
    int ub = Math.max(q.vertexSet().size(), g.vertexSet().size())
        + q.edgeSet().size() + g.edgeSet().size();
    intBound = Math.min(intBound, ub);

    // convert to integer labels
    // TODO: how to best do that... if there are no labels..
    V vtest = null;
    E etest = null;
    if (!g.vertexSet().isEmpty())
    {
      vtest = g.vertexSet().iterator().next();
    }
    else if (!q.vertexSet().isEmpty())
    {
      vtest = q.vertexSet().iterator().next();
    }

    if (!g.edgeSet().isEmpty())
    {
      etest = g.edgeSet().iterator().next();
    }
    else if (!q.edgeSet().isEmpty())
    {
      etest = q.edgeSet().iterator().next();
    }
    if (vtest == null)
    {
      System.err.println("No vertices in graphs");
    }
    if (etest == null)
    {
      System.err.println("No edges in graphs");
    }

    HashMap<V, Integer> va = new HashMap<V, Integer>();
    HashMap<E, Integer> ea = new HashMap<E, Integer>();
    if (vtest instanceof LabeledVertex)
    {
      HashMap<String, Integer> labelMap = new HashMap<String, Integer>();
      int nextkey = 0;
      for (V v : g.vertexSet())
      {
        if (!labelMap.containsKey(v.toString()))
        {
          labelMap.put(v.toString(), nextkey);
          nextkey++;
        }
        va.put(v, labelMap.get(v.toString()));
      }
      for (V v : q.vertexSet())
      {
        if (!labelMap.containsKey(v.toString()))
        {
          labelMap.put(v.toString(), nextkey);
          nextkey++;
        }
        va.put(v, labelMap.get(v.toString()));
      }
    }
    else
    {
      for (V v : g.vertexSet()) // no labels = uniform labels
      {
        va.put(v, 0);
      }
      for (V v : q.vertexSet()) // no labels = uniform labels
      {
        va.put(v, 0);
      }
    }
    if (etest instanceof LabeledEdge)
    {
      HashMap<String, Integer> labelMap = new HashMap<String, Integer>();
      int nextkey = 0;
      for (E e : g.edgeSet())
      {
        if (!labelMap.containsKey(e.toString()))
        {
          labelMap.put(e.toString(), nextkey);
          nextkey++;
        }
        ea.put(e, labelMap.get(e.toString()));
      }
      for (E e : q.edgeSet())
      {
        if (!labelMap.containsKey(e.toString()))
        {
          labelMap.put(e.toString(), nextkey);
          nextkey++;
        }
        ea.put(e, labelMap.get(e.toString()));
      }
    }
    else
    {
      for (E e : g.edgeSet()) // no labels = uniform labels
      {
        ea.put(e, 0);
      }
      for (E e : q.edgeSet()) // no labels = uniform labels
      {
        ea.put(e, 0);
      }
    }

    Pair<int[], int[][]> Q = graphToPrimitiveDFS(g, va, ea);
    Pair<int[], int[][]> G = graphToPrimitiveDFS(q, va, ea);

    return compute(Q, G, intBound);
  }

  private double compute(Pair<int[], int[][]> Q, Pair<int[], int[][]> G,
      int bound)
  {

    BSEditDistanceJNI bsed = new BSEditDistanceJNI();
    int result = bsed.getEditDistance(width, Q.getFirst(), Q.getSecond(),
        G.getFirst(), G.getSecond(), bound);
    return result;
  }

  private <V extends Vertex, E extends Edge> Pair<int[], int[][]> graphToPrimitiveDFS(
      AbstractGraph<V, E> g, HashMap<V, Integer> va, HashMap<E, Integer> ea)
  {

    HashMap<V, Integer> rank = new HashMap<V, Integer>();

    int count = 0;
    ArrayList<V> sortedbyDegree = new ArrayList<V>(g.vertexSet());
    Collections.sort(sortedbyDegree, new Comparator<V>()
    {
      @Override
      public int compare(V v1, V v2)
      {
        return g.degreeOf(v1) - g.degreeOf(v2);
      }
    });

    for (V v : sortedbyDegree)
    {
      if (rank.get(v) == null)
      { // unvisited
        count = doDFS(g, rank, v, count);
      }
    }

    int[] vertices = new int[g.vertexSet().size()];
    int[][] edges = new int[g.edgeSet().size()][];

    for (V v : sortedbyDegree)
    {
      vertices[rank.get(v)] = va.get(v);
    }
    int edgeIndex = 0;
    for (E e : g.edgeSet())
    {

      int iU = rank.get(g.getEdgeSource(e));
      int iV = rank.get(g.getEdgeTarget(e));
      int l = ea.get(e);
      edges[edgeIndex] = new int[] { iU, iV, l};
      edgeIndex++;
    }

    return new Pair<int[], int[][]>(vertices, edges);
  }

  private <V extends Vertex, E extends Edge> int doDFS(AbstractGraph<V, E> g,
      HashMap<V, Integer> rank, V v, int count)
  {
    rank.put(v, count++);
    for (E e : g.edgesOf(v))
    {
      V w = g.getEdgeSource(e);
      if (v == w)
      {
        w = g.getEdgeTarget(e);
      }
      if (rank.get(w) == null)
      {// unvisited
        count = doDFS(g, rank, w, count);
      }
    }
    return count;
  }
}
