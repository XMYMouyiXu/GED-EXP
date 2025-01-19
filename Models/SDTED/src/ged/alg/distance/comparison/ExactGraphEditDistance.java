package ged.alg.distance.comparison;

import java.util.ArrayList;

import org.jgrapht.graph.AbstractGraph;

import ged.alg.distance.GraphDistance;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;

public class ExactGraphEditDistance extends GraphDistance
{
  @Override
  public String getID()
  {
    return "EXACTGED";
  }

  /**
   * Do not use, naive implementation.
   * Might terminate early, if ged=0
   *
   * @param g1
   * @param g2
   * @return the exact graph edit distance of g1 and g2, based on trying all
   *         edit paths and choosing the minimum.
   */
  @Override
  public <V extends Vertex, E extends Edge> double computeGraphDistance(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    // choose bigger graph to be g1
    if (g1.vertexSet().size() < g2.vertexSet().size())
    {
      AbstractGraph<V, E> temp = g1;
      g1 = g2;
      g2 = temp;
    }
    // lists for vertex ordering
    ArrayList<V> vertices1 = new ArrayList<V>(g1.vertexSet());
    ArrayList<V> vertices2 = new ArrayList<V>(g2.vertexSet());

    int vertices1Size = vertices1.size();

    // Try all possible combinations
    int[] indices = new int[vertices1Size];

    int[] assignment = new int[vertices1Size];
    for (int i = 0; i < vertices1Size; i++)
      assignment[i] = i;

    int i = 0;
    double min = computeCostOfEditPath(g1, g2, vertices1, vertices2,
        assignment); // This combination is not computed afterwards
    if (min == 0) //
      return min;
    while (i < vertices1Size)
    {
      if (indices[i] < i)
      {
        swap(assignment, i % 2 == 0 ? 0 : indices[i], i);
        double cost = computeCostOfEditPath(g1, g2, vertices1, vertices2,
            assignment);
        if (cost < min)
        {
          min = cost;
          if (min == 0)//
            return min;
        }
        indices[i]++;
        i = 0;
      }
      else
      {
        indices[i] = 0;
        i++;
      }
    }
    return min;
  }

  /**
   * Swaps the elements of input at positions a and b
   *
   * @param input
   * @param a
   * @param b
   */
  private static void swap(int[] input, int a, int b)
  {
    int tmp = input[a];
    input[a] = input[b];
    input[b] = tmp;
  }

}
