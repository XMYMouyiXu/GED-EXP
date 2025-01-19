package ged.alg.graphencoder.tree;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.NotImplementedException;
import org.jgrapht.graph.AbstractGraph;
import org.jgrapht.graph.DefaultDirectedGraph;

import ged.alg.graphencoder.GraphEncoder;
import ged.structures.graph.Edge;
import ged.structures.graph.VertexWrapper;
import ged.structures.tree.CTree;

public class NeighborhoodTreeWL<V, E extends Edge> extends GraphEncoder<V, E>
{
  private final List<CTree<VertexWrapper<V>, Edge>> trees = new ArrayList<CTree<VertexWrapper<V>, Edge>>();

  public NeighborhoodTreeWL(AbstractGraph<V, E> graph)
  {
    super(graph);
    for (V v : graph.vertexSet())
      trees.add(new CTree<VertexWrapper<V>, Edge>(
          new DefaultDirectedGraph<VertexWrapper<V>, Edge>(Edge.class),
          new VertexWrapper<V>(v)));
  }

  public NeighborhoodTreeWL(AbstractGraph<V, E> graph, int steps)
  {
    this(graph);
    refinementSteps(steps);
  }

  public List<CTree<VertexWrapper<V>, Edge>> getCurrentTreeRepresentations()
  {
    return trees;
  }

  @Override
  public boolean refine()
  {
    if (trees.isEmpty()) return false; // This is only possible if graph has no
    // vertices

    for (CTree<VertexWrapper<V>, Edge> tree : trees)
      refineTree(tree);

    return !trees.isEmpty();
  }

  private boolean refineTree(CTree<VertexWrapper<V>, Edge> tree)
  {
    int curMaxLayer = tree.getMaxLayer();
    Collection<VertexWrapper<V>> leaves = tree
        .getVerticesAtDistance(curMaxLayer);
    if (leaves == null)// safety check should not happen (except no root)
      return false;

    Map<V, VertexWrapper<V>> childMap = new HashMap<V, VertexWrapper<V>>();
    // We can save space by adding the each node at most once each layer
    for (VertexWrapper<V> leaf : leaves)
    {
      // find neighbors to extend the tree
      for (E e : graph.edgesOf(leaf.getObject()))
      {
        V u = graph.getEdgeSource(e);
        if (u.equals(leaf.getObject())) u = graph.getEdgeTarget(e);

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
    // did we get a new layer? then our tree was refined (should be true every
    // time with WL
    return tree.getMaxLayer() == curMaxLayer + 1;
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

  @Override
  public void fullyRefine()
  {
    throw new NotImplementedException(
        "This WL implementation cannot automatically fully refine.");
  }

}
