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
import ged.structures.tree.CTree;

public class NeighborhoodTreeV1<V, E extends Edge> extends GraphEncoder<V, E>
{
  private final List<CTree<V, Edge>> trees = new ArrayList<CTree<V, Edge>>();
  private final HashSet<CTree<V, Edge>> unrefinedTrees;

  public NeighborhoodTreeV1(AbstractGraph<V, E> graph)
  {
    super(graph);
    for (V v : graph.vertexSet())
      trees.add(
          new CTree<V, Edge>(new DefaultDirectedGraph<V, Edge>(Edge.class), v));
    unrefinedTrees = new HashSet<CTree<V, Edge>>(trees);
  }

  public NeighborhoodTreeV1(AbstractGraph<V, E> graph, boolean refine)
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
  public NeighborhoodTreeV1(AbstractGraph<V, E> graph, int refinementsteps)
  {
    this(graph);
    refinementSteps(refinementsteps);
  }

  public List<CTree<V, Edge>> getCurrentTreeRepresentations()
  {
    return trees;
  }

  @Override
  public boolean refine()
  {
    if (unrefinedTrees.isEmpty()) return false;

    List<CTree<V, Edge>> doneTrees = new ArrayList<CTree<V, Edge>>();
    for (CTree<V, Edge> tree : unrefinedTrees)
      if (!refineTree(tree)) doneTrees.add(tree);

    unrefinedTrees.removeAll(doneTrees);

    return !unrefinedTrees.isEmpty();
  }

  private boolean refineTree(CTree<V, Edge> tree)
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
    return tree.getMaxLayer() == nextLayer;
  }

  @Override
  public String computeLexicographicEncoding()
  {
    List<String> vertexEncodings = new ArrayList<String>();
    for (CTree<V, Edge> tree : trees)
      vertexEncodings.add(tree.computeLexicographicEncoding());
    Collections.sort(vertexEncodings);

    StringBuilder encoding = new StringBuilder();
    for (String childEncoding : vertexEncodings)
      encoding.append(childEncoding).append("\n");

    return encoding.toString();
  }
}
