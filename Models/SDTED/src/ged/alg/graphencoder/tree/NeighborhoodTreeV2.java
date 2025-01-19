package ged.alg.graphencoder.tree;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.jgrapht.graph.AbstractGraph;
import org.jgrapht.graph.DefaultDirectedGraph;

import ged.alg.graphencoder.GraphEncoder;
import ged.structures.graph.Edge;
import ged.structures.graph.VertexWrapper;
import ged.structures.tree.CTree;

public class NeighborhoodTreeV2<V, E extends Edge> extends GraphEncoder<V, E>
{
  private final List<CTree<VertexWrapper<V>, Edge>> trees = new ArrayList<CTree<VertexWrapper<V>, Edge>>();
  private final Map<CTree<VertexWrapper<V>, Edge>, HashMap<V, Integer>> depthMaps = new HashMap<CTree<VertexWrapper<V>, Edge>, HashMap<V, Integer>>();
  private final HashSet<CTree<VertexWrapper<V>, Edge>> unrefinedTrees;

  public NeighborhoodTreeV2(AbstractGraph<V, E> graph)
  {
    super(graph);
    for (V v : graph.vertexSet())
    {
      CTree<VertexWrapper<V>, Edge> tree = new CTree<VertexWrapper<V>, Edge>(
          new DefaultDirectedGraph<VertexWrapper<V>, Edge>(Edge.class),
          new VertexWrapper<V>(v));
      trees.add(tree);
      HashMap<V, Integer> depthMap = new HashMap<V, Integer>();
      depthMap.put(v, 0);
      depthMaps.put(tree, depthMap);
    }
    unrefinedTrees = new HashSet<CTree<VertexWrapper<V>, Edge>>(trees);
  }

  public NeighborhoodTreeV2(AbstractGraph<V, E> graph, boolean refine)
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
  public NeighborhoodTreeV2(AbstractGraph<V, E> graph, int refinementsteps)
  {
    this(graph);
    refinementSteps(refinementsteps);
  }

  public List<CTree<VertexWrapper<V>, Edge>> getCurrentTreeRepresentations()
  {
    return trees;
  }

  @Override
  public boolean refine()
  {
    if (unrefinedTrees.isEmpty()) return false;

    List<CTree<VertexWrapper<V>, Edge>> doneTrees = new ArrayList<CTree<VertexWrapper<V>, Edge>>();

    for (CTree<VertexWrapper<V>, Edge> tree : unrefinedTrees)
      if (!refineTree(tree, depthMaps.get(tree))) doneTrees.add(tree);

    unrefinedTrees.removeAll(doneTrees);

    return !unrefinedTrees.isEmpty();
  }

  private boolean refineTree(CTree<VertexWrapper<V>, Edge> tree,
      HashMap<V, Integer> depthMap)
  {
    int curMaxLayer = tree.getMaxLayer();
    Collection<VertexWrapper<V>> leaves = tree
        .getVerticesAtDistance(curMaxLayer);
    if (leaves == null)// safety check should not happen (except no root)
      return false;

    int nextLayer = curMaxLayer + 1;
    Map<V, VertexWrapper<V>> childMap = new HashMap<V, VertexWrapper<V>>();

    for (VertexWrapper<V> leaf : leaves)
    {
      V v = leaf.getObject();
      // find new neighbors to extend the tree
      for (E e : graph.edgesOf(v))
      {
        V u = graph.getEdgeSource(e);
        if (u.equals(v)) u = graph.getEdgeTarget(e);

        Integer distanceToRoot = depthMap.get(u);
        if (distanceToRoot == null)
        {
          VertexWrapper<V> newChild = new VertexWrapper<V>(u);
          tree.addNode(newChild, leaf, e.copy());
          childMap.put(u, newChild);
          depthMap.put(u, nextLayer);
        }
        else if (distanceToRoot == nextLayer || distanceToRoot == curMaxLayer)
        {
          VertexWrapper<V> childInTree = childMap.get(u);

          if (childInTree == null)
          {
            VertexWrapper<V> newChild = new VertexWrapper<V>(u);
            tree.addNode(newChild, leaf, e.copy());
            childMap.put(u, newChild);
          }
          else
            tree.addParent(childInTree, leaf, e.copy());
        }
      }
    }
    // did we get a new layer? then our tree was refined
    return tree.getMaxLayer() == nextLayer;
  }

  @Override
  public String computeLexicographicEncoding()
  {
    List<String> vertexEncodings = new ArrayList<String>();
    for (CTree<VertexWrapper<V>, Edge> tree : trees)
      vertexEncodings.add(tree.computeLexicographicEncoding());
    Collections.sort(vertexEncodings);

    StringBuilder encoding = new StringBuilder();
    for (String childEncoding : vertexEncodings)
      encoding.append(childEncoding).append("\n");

    return encoding.toString();
  }

}
