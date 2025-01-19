package ged.alg.graphencoder.tree;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import org.jgrapht.graph.AbstractGraph;
import org.jgrapht.graph.DefaultDirectedGraph;

import ged.alg.graphencoder.GraphEncoder;
import ged.structures.graph.Edge;
import ged.structures.tree.CTreePlus;

public class NeighborhoodTreeV1Plus<V, E extends Edge>
extends
GraphEncoder<V, E>
{
  private final List<CTreePlus<V, Edge>> trees = new ArrayList<CTreePlus<V, Edge>>();
  private final HashSet<CTreePlus<V, Edge>> unrefinedTrees;

  public NeighborhoodTreeV1Plus(AbstractGraph<V, E> graph)
  {
    super(graph);
    for (V v : graph.vertexSet())
      trees.add(new CTreePlus<V, Edge>(
          new DefaultDirectedGraph<V, Edge>(Edge.class), v));
    unrefinedTrees = new HashSet<CTreePlus<V, Edge>>(trees);
  }

  public NeighborhoodTreeV1Plus(AbstractGraph<V, E> graph, boolean refine)
  {
    this(graph);
    if (refine) fullyRefine();
  }

  /**
   * @param graph
   * @param refinementsteps
   *          maximum number of refinement steps (possibly less if graph is
   *          smaller), <0: fully refine
   */
  public NeighborhoodTreeV1Plus(AbstractGraph<V, E> graph, int refinementsteps)
  {
    this(graph);
    refinementSteps(refinementsteps);
  }

  public List<CTreePlus<V, Edge>> getCurrentTreeRepresentations()
  {
    return trees;
  }

  @Override
  public boolean refine()
  {
    if (unrefinedTrees.isEmpty()) return false;

    List<CTreePlus<V, Edge>> doneTrees = new ArrayList<CTreePlus<V, Edge>>();
    for (CTreePlus<V, Edge> tree : unrefinedTrees)
      if (!refineTree(tree)) doneTrees.add(tree);

    unrefinedTrees.removeAll(doneTrees);

    return !unrefinedTrees.isEmpty();
  }

  private boolean refineTree(CTreePlus<V, Edge> tree)
  {
    int curMaxLayer = tree.getMaxLayer();
    Collection<V> leaves = tree.getVerticesAtDistance(curMaxLayer);
    if (leaves == null)// safety check should not happen (except no root)
      return false;

    int nextLayer = curMaxLayer + 1;
    for (V leaf : leaves)
    {
      // find new neighbors to extend the tree
      for (E e : graph.edgesOf(leaf))
      {
        V u = graph.getEdgeSource(e);
        if (u.equals(leaf)) u = graph.getEdgeTarget(e);

        Integer distanceToRoot = tree.getDistanceToRootFor(u);
        if (distanceToRoot == null)
          tree.addNode(u, leaf, e.copy());
        else if (distanceToRoot == nextLayer)// already in tree
          tree.addParent(u, leaf, e.copy());// only update parent
      }
    }
    // did we get a new layer? then our tree was refined
    if (tree.getMaxLayer() != nextLayer)//
      return false;

    tree.buildLastLayerCliques(graph);
    return true;
  }

  @Override
  public String computeLexicographicEncoding()
  {
    List<String> vertexEncodings = new ArrayList<String>();
    for (CTreePlus<V, Edge> tree : trees)
      vertexEncodings.add(tree.computeLexicographicEncoding());
    Collections.sort(vertexEncodings);

    StringBuilder encoding = new StringBuilder();
    for (String childEncoding : vertexEncodings)
      encoding.append(childEncoding).append("\n");

    return encoding.toString();
  }

}
