package ged.structures.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.jgrapht.graph.AbstractBaseGraph;
import org.jgrapht.graph.AbstractGraph;

/**
 * Compressed Tree This represents a tree with some subgraphs at the same level
 * being the same. Storing this information in the tree allows more efficient
 * computation as less redundant computations have to be performed
 */
public class CTreePlus<V, E> extends Tree<V, E>
{
  private final Comparator<? super Entry<String, List<V>>> comp = (e1, e2) -> {
    int stringCompare = e1.getKey().compareTo(e2.getKey());
    if (stringCompare != 0) return stringCompare;
    return e1.getValue().size() - e2.getValue().size();
  };

  /**
   * note that these are not cliques in the usual sense (not every v is
   * connected to every v here). We define a clique here as vertices that can
   * reach each other directly or indirectly while only passing nodes on the
   * same layer of this tree.
   */
  private final HashMap<V, HashSet<V>> cliques = new HashMap<V, HashSet<V>>();

  public CTreePlus(AbstractBaseGraph<V, E> graph)
  {
    super(graph);
  }

  public CTreePlus(AbstractBaseGraph<V, E> graph, V v)
  {
    super(graph);
    addNode(v);
    initClique(v);
  }

  public void addNode(V vertex, V[] parents, E[] edges)
  {
    if (vertex == null) return;
    if (parents == null)
    {
      addNode(vertex);
      return;
    }
    if (graph.containsVertex(vertex)) throw new IllegalArgumentException(
        "Tree already contains a vertex with the identifier: "
            + vertex.toString());
    if (parents.length == 0) throw new IllegalArgumentException(
        "Every non-root vertex needs at least one parent");

    int parentLayer = vertexToRootDistance.get(parents[0]);
    for (int i = 1; i < parents.length; ++i)
      if (vertexToRootDistance.get(parents[i]) != parentLayer)
        throw new IllegalArgumentException(
            "Parents of a vertex have to be on the same layer!");

    graph.addVertex(vertex);
    for (int i = 0; i < parents.length; ++i)
      graph.addEdge(parents[i], vertex, edges != null ? edges[i] : null);

    store(vertex, vertexToRootDistance.get(parents[0]) + 1);
  }

  public List<V> getParents(V child)
  {
    List<V> parents = new ArrayList<V>();
    for (E edge : graph.edgesOf(child))
    {
      V potentialParent = graph.getEdgeSource(edge);
      if (potentialParent != child) parents.add(potentialParent);
    }
    return parents;
  }

  public void addParent(V child, V parent, E edge)
  {
    if (child == null) return;
    if (parent == null) return;
    if (!graph.containsVertex(child)) throw new IllegalArgumentException(
        "Tree does not contain child: " + child.toString());
    if (!graph.containsVertex(parent)) throw new IllegalArgumentException(
        "Tree does not contain parent: " + parent.toString());
    if (graph.containsEdge(parent, child) || graph.containsEdge(child, parent))
      throw new IllegalArgumentException("Tree already contains edge: "
          + parent.toString() + " from/to " + child.toString());
    if (edge == null)
      graph.addEdge(parent, child);
    else
      graph.addEdge(parent, child, edge);
  }

  @Override
  public String computeLexicographicEncoding()
  {
    Map<V, Integer> variableMap = new HashMap<>();
    StringBuilder[] layerEncoding = new StringBuilder[maxLayer + 1];

    for (int i = maxLayer; i >= 0; --i)
    {
      Map<String, List<V>> encodingVertexMap = new HashMap<String, List<V>>();

      for (V leaf : getVerticesAtDistance(i))
      {
        StringBuilder vertexEncodingBuilder = new StringBuilder(
            leaf.toString());
        Set<E> childEdges = getChildEdges(leaf);
        int numChildren = childEdges.size();
        if (numChildren > 0)
        {
          String[] childEncodings = new String[numChildren];
          int c = 0;
          for (E childEdge : childEdges)
            childEncodings[c++] = "(" + childEdge + ")"
                + variableMap.get(graph.getEdgeTarget(childEdge));
          Arrays.sort(childEncodings);

          vertexEncodingBuilder.append("[");
          for (String encoding : childEncodings)
            vertexEncodingBuilder.append(encoding);
          vertexEncodingBuilder.append("]");
        }

        String vertexEncoding = vertexEncodingBuilder.toString();
        List<V> verticesWithEncoding = encodingVertexMap.get(vertexEncoding);
        if (verticesWithEncoding == null) encodingVertexMap.put(vertexEncoding,
            verticesWithEncoding = new ArrayList<V>());
        verticesWithEncoding.add(leaf);
      }
      List<Entry<String, List<V>>> entries = new ArrayList<Entry<String, List<V>>>(
          encodingVertexMap.entrySet());
      entries.sort(comp);

      StringBuilder layerEncodingString = new StringBuilder();
      int numEntries = entries.size();
      for (int t = 0; t < numEntries; ++t)
      {
        Entry<String, List<V>> entry = entries.get(t);
        layerEncodingString.append(" ").append(t).append(":(")
        .append(entry.getKey()).append(")");
        // + collectedVertices.size()); //should we add the number of equivalent
        // subtrees?
        HashMap<HashSet<V>, Integer> cliqueSizes = new HashMap<HashSet<V>, Integer>();

        for (V vertex : entry.getValue())
        {
          variableMap.put(vertex, t);
          cliqueSizes.computeIfAbsent(cliques.get(vertex), HashSet::size);
        }

        // add the clique size(s) information to disambiguate some cases
        List<Integer> cliqueValues = new ArrayList<Integer>(
            cliqueSizes.values());
        Collections.sort(cliqueValues);
        layerEncodingString.append(cliqueValues);
      }

      layerEncoding[i] = layerEncodingString;
    }

    StringBuilder encoding = new StringBuilder();

    for (int i = 0; i <= maxLayer; ++i)
      encoding.append("L" + i).append(layerEncoding[i]).append("\n");
    return encoding.toString();
  }

  public void buildLastLayerCliques(AbstractGraph<V, ?> completeGraph)
  {
    Collection<V> vertices = getVerticesAtDistance(maxLayer);
    for (V v : vertices)
      initClique(v);
    for (V v : vertices)
      collectClique(v, vertices, completeGraph);
  }

  private void initClique(V v)
  {
    HashSet<V> clique = new HashSet<V>();
    clique.add(v);
    cliques.put(v, clique);
  }

  private <E_> void collectClique(V v, Collection<V> vertices,
      AbstractGraph<V, E_> completeGraph)
  {
    HashSet<V> clique = cliques.get(v);

    for (E_ e : completeGraph.edgesOf(v))
    {
      V neighbor = completeGraph.getEdgeSource(e);
      if (neighbor == v) neighbor = completeGraph.getEdgeTarget(e);

      if (vertices.contains(neighbor) && !clique.contains(neighbor))//
        for (V other : cliques.get(neighbor))//
          if (clique.add(other))//
            cliques.put(other, clique);
    }
  }
}
