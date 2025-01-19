package ged.structures.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.jgrapht.graph.AbstractBaseGraph;
import org.jgrapht.graph.builder.GraphTypeBuilder;

/**
 * Compressed Tree This represents a tree with some subgraphs at the same level
 * being the same. Storing this information in the tree allows more efficient
 * computation as less redundant computations have to be performed
 */
public class CTree<V, E> extends Tree<V, E>
{
  private final Comparator<? super Entry<String, List<V>>> comp = (e1, e2) -> {
    int stringCompare = e1.getKey().compareTo(e2.getKey());
    if (stringCompare != 0) return stringCompare;
    return e1.getValue().size() - e2.getValue().size();
  };

  public CTree(AbstractBaseGraph<V, E> graph)
  {
    super(graph);
  }

  public CTree(AbstractBaseGraph<V, E> graph, V v)
  {
    super(graph);
    addNode(v);
  }

  @SuppressWarnings("unchecked")
  public void addNode(V vertex, V... parents)
  {
    addNode(vertex, parents, null);
  }


  private void addNode(V vertex, List<V> parents, List<E> edges)
  {
    if (vertex == null) return;
    if (parents == null)
    {
      addNode(vertex);
      return;
    }
    int numParents = parents.size();
    if (numParents == 0) throw new IllegalArgumentException(
        "Every non-root vertex needs at least one parent");

    int parentLayer = vertexToRootDistance.get(parents.get(0));
    for (int i = 1; i < numParents; ++i)
      if (vertexToRootDistance.get(parents.get(i)) != parentLayer)
        throw new IllegalArgumentException(
            "Parents of a vertex have to be on the same layer!");

    if (!graph.addVertex(vertex))// already contained check
      throw new IllegalArgumentException(
          "Tree already contains a vertex with the identifier: "
              + vertex.toString());

    if (edges == null)
      for (V parent : parents)
        graph.addEdge(parent, vertex);
    else
      for (int i = 0; i < numParents; ++i)
        graph.addEdge(parents.get(i), vertex, edges.get(i));

    store(vertex, parentLayer + 1);
  }

  public void addNode(V vertex, V[] parents, E[] edges)
  {
    if (vertex == null) return;
    if (parents == null)
    {
      addNode(vertex);
      return;
    }
    if (parents.length == 0) throw new IllegalArgumentException(
        "Every non-root vertex needs at least one parent");

    int parentLayer = vertexToRootDistance.get(parents[0]);
    for (int i = 1; i < parents.length; ++i)
      if (vertexToRootDistance.get(parents[i]) != parentLayer)
        throw new IllegalArgumentException(
            "Parents of a vertex have to be on the same layer!");

    if (!graph.addVertex(vertex))// already contained check
      throw new IllegalArgumentException(
          "Tree already contains a vertex with the identifier: "
              + vertex.toString());

    if (edges == null)
      for (V parent : parents)
        graph.addEdge(parent, vertex);
    else
      for (int i = 0; i < parents.length; ++i)
        graph.addEdge(parents[i], vertex, edges[i]);

    store(vertex, parentLayer + 1);
  }

  public List<V> getParents(V child)
  {
    Set<E> incommingEdges = graph.incomingEdgesOf(child);
    List<V> parents = new ArrayList<V>(incommingEdges.size());
    for (E edge : incommingEdges)
      parents.add(graph.getEdgeSource(edge));
    return parents;
  }

  public void addParent(V child, V parent, E edge)
  {
    Integer parentDistance = vertexToRootDistance.get(parent);
    if (parentDistance == null
        || parentDistance + 1 != vertexToRootDistance.get(child))
      throw new IllegalArgumentException(
          "Cannot add an edge between nodes as the parent is not one layer lower than child.");

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
        for (V vertex : entry.getValue())
          variableMap.put(vertex, t);
      }
      layerEncoding[i] = layerEncodingString;
    }

    StringBuilder encoding = new StringBuilder();

    for (int i = 0; i <= maxLayer; ++i)
      encoding.append("L").append(i).append(layerEncoding[i]).append("\n");
    return encoding.toString();
  }

  private CTree<V, E> subTreeAt(V vertex)
  {
    CTree<V, E> subTree = new CTree<V, E>(
        (AbstractBaseGraph<V, E>) GraphTypeBuilder.forGraph(graph)
        .buildGraph(),
        vertex);

    Integer newRootDistance = vertexToRootDistance.get(vertex);
    if (newRootDistance == null)//
      return subTree;

    HashSet<V> previousLayer = new HashSet<>();
    previousLayer.add(vertex);

    int numLayers = getMaxLayer() - newRootDistance;
    for (int i = 0; i < numLayers; ++i)
    {
      HashSet<V> nextLayer = new HashSet<V>();
      for (V previousVertex : previousLayer)
        for (E outEdge : graph.outgoingEdgesOf(previousVertex))
          nextLayer.add(graph.getEdgeTarget(outEdge));

      if (nextLayer.isEmpty())//
        break;

      for (V layerVertex : nextLayer)
      {
        List<V> parents = getParents(layerVertex);
        List<V> includedParents = new ArrayList<>(parents.size());
        for (V parent : parents)
          if (previousLayer.contains(parent))//
            includedParents.add(parent);
        List<E> includedEdges = new ArrayList<>(includedParents.size());
        for (V parent : includedParents)
          includedEdges.add(graph.getEdge(parent, layerVertex));
        subTree.addNode(layerVertex, includedParents, includedEdges);
      }
      previousLayer = nextLayer;
    }

    return subTree;
  }



}
