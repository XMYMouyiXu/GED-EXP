package ged.alg.distance.comparison;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.jgrapht.graph.AbstractGraph;

import com.google.common.collect.Lists;

import ged.alg.assignment.HungarianAlgorithm;
import ged.alg.distance.GraphDistance;
import ged.alg.distance.comparison.beamsearch.BeamSearch;
import ged.structures.graph.Edge;
import ged.structures.graph.LabeledEdge;
import ged.structures.graph.PGraph;
import ged.structures.graph.Vertex;

public class SubgraphMatching extends GraphDistance
{
  @Override
  public String getID()
  {
    return "SubgraphMatching_"+k;
  }
  private int k;
  private BeamSearch beam;
  public SubgraphMatching(int k)
  {
	  this.k = k;
	  this.beam = new BeamSearch(1000);
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

    HashMap<V,AbstractGraph<V, E>> graphlets = getGraphlets(g1,vertices1,g2,vertices2);
    // vertex matching (upper left corner)
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < m; j++)
      {
        V v1 = vertices1.get(i);
        V v2 = vertices2.get(j);
        C[i][j] = vertexSubstitution(v1, g1, graphlets.get(v1), v2, g2, graphlets.get(v2));
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
      C[n + j][j] = vertexInsertionDeletion(graphlets.get(v2));
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
      C[i][m + i] = vertexInsertionDeletion(graphlets.get(v1));
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

  private <V extends Vertex, E extends Edge> HashMap<V, AbstractGraph<V, E>> getGraphlets(AbstractGraph<V, E> g1, List<V> vertices1,
		AbstractGraph<V, E> g2, List<V> vertices2) {
	  HashMap<V,AbstractGraph<V, E>> graphlets = new HashMap<V,AbstractGraph<V, E>>();
	  for(V v1: vertices1)
	  {
		  graphlets.put(v1,computeGraphlet(g1,v1));
	  }
	  for(V v2: vertices2)
	  {
		  graphlets.put(v2,computeGraphlet(g2,v2));
	  }
	  return graphlets;
}

private <V extends Vertex, E extends Edge> AbstractGraph<V, E> computeGraphlet(AbstractGraph<V, E> g1, V v1) {
	PGraph<V, E> graphlet = new PGraph<V, E>((Class<? extends E>) LabeledEdge.class);
	HashSet<V> verticesAtK = new HashSet<V>();
	graphlet.addVertex(v1);
	verticesAtK.add(v1);
	for(int i=0; i<this.k; i++)
	{
		HashSet<V> verticesAtKplus = new HashSet<V>();
		for(V v: verticesAtK)
		{
			for(E e: g1.edgesOf(v))
			{
				V u = g1.getEdgeSource(e);
				if(u.equals(v))
				{
					u = g1.getEdgeTarget(e);
				}
				if(!graphlet.containsVertex(u))
				{
					graphlet.addVertex(u);
					verticesAtKplus.add(u);
				}
				if(!graphlet.containsEdge(e))
				{
					graphlet.addEdge(v, u, e);
				}
			}
		}
		verticesAtK = verticesAtKplus;
	}
			
	return (AbstractGraph<V, E>) graphlet;
}

private <V extends Vertex, E extends Edge> double vertexInsertionDeletion(AbstractGraph<V, E> g)
  {
    double rV = 0;
    double rE = 0;
    for(V v : g.vertexSet())
    {
    	rV += v.getDeletionCost();
    }
    // add costs for all edge deletions
    for (E e : g.edgeSet())
    {
      rE += e.getDeletionCost();
    }

    return rV + rE;
  }

  private <V extends Vertex, E extends Edge> double vertexSubstitution(
      V v1, AbstractGraph<V, E> g1, AbstractGraph<V, E> graphlet1, V v2, AbstractGraph<V, E> g2,AbstractGraph<V, E> graphlet2)
  {
    return this.beam.getEditDistance(graphlet1, graphlet2,v1,v2);
  }

  private static final int[] assignment(double[][] C)
  {
    return HungarianAlgorithm.assignment(C);
  }

}
