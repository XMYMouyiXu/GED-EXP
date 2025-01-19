package ged.structures.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.jgrapht.graph.AbstractBaseGraph;

public class Tree<V, E>
{
  protected final Map<V, Integer> vertexToRootDistance = new HashMap<V, Integer>();
  protected final Map<Integer, Collection<V>> rootDistanceToVertices = new HashMap<Integer, Collection<V>>();
  protected int maxLayer = -1;
  private final Map<V, String> encodingCache = new HashMap<>();
  protected final AbstractBaseGraph<V, E> graph;
  protected V root;

  public static final class ChildWithEdge<V, E>
  {
    public final V v;
    public final E e;
    public ChildWithEdge(V v, E e)
    {
      this.v = v;
      this.e = e;
    }
  }

  public Tree(AbstractBaseGraph<V, E> graph)
  {
    this.graph = graph;
  }

  public Tree(AbstractBaseGraph<V, E> graph, V root)
  {
    this(graph);
    addNode(root);
  }

  public void addNode(V vertex)
  {
    addNode(vertex, null, null);
  }

  public void addNode(V vertex, V parent)
  {
    addNode(vertex, parent, null);
  }

  public void addNode(V vertex, V parent, E edge)
  {
    if (vertex == null) return;
    if (parent == null)
    {
      if (root != null) throw new IllegalArgumentException(
          "There already exists a root node, please define the parent!");

      root = vertex;
      graph.addVertex(vertex);
      store(vertex, 0);
    }
    else
    {
      if (!graph.addVertex(vertex))// already contained check
        throw new IllegalArgumentException(
            "Tree already contains a vertex with the identifier: "
                + vertex.toString());

      if (edge == null)
        graph.addEdge(parent, vertex);
      else
        graph.addEdge(parent, vertex, edge);

      store(vertex, vertexToRootDistance.get(parent) + 1);
    }
  }

  protected void store(V vertex, int layer)
  {
    vertexToRootDistance.put(vertex, layer);
    getOrCreateLayer(layer).add(vertex);
  }

  private Collection<V> getOrCreateLayer(int layer)
  {
    Collection<V> vertices = rootDistanceToVertices.get(layer);
    if (vertices == null)
      rootDistanceToVertices.put(layer, vertices = new HashSet<V>());
    if (layer > maxLayer) maxLayer = layer;
    return vertices;
  }

  public int getMaxLayer()
  {
    return maxLayer;
  }

  public Collection<V> getVerticesAtDistance(int edgesToRoot)
  {
    Collection<V> vertices = rootDistanceToVertices.get(edgesToRoot);
    return vertices == null ? null
        : Collections.unmodifiableCollection(vertices);
  }

  public Integer getDistanceToRootFor(V vertex)
  {
    return vertexToRootDistance.get(vertex);
  }

  public final V getRoot()
  {
    return root;
  }

  public V getParent(V child)
  {
    for (E edge : graph.incomingEdgesOf(child))
      return graph.getEdgeSource(edge);
    return null;
  }

  public List<V> getChildren(V parent)
  {
    Set<E> outgoingEdges = graph.outgoingEdgesOf(parent);
    List<V> children = new ArrayList<V>(outgoingEdges.size());
    for (E edge : outgoingEdges)
      children.add(graph.getEdgeTarget(edge));
    return children;
  }

  public Set<E> getChildEdges(V parent)
  {
    return graph.outgoingEdgesOf(parent);
  }

  public List<ChildWithEdge<V, E>> getChildrenWithEdges(V parent)
  {
    Set<E> outgoingEdges = graph.outgoingEdgesOf(parent);
    List<ChildWithEdge<V, E>> children = new ArrayList<>(outgoingEdges.size());
    for (E edge : outgoingEdges)
      children.add(new ChildWithEdge<V, E>(graph.getEdgeTarget(edge), edge));
    return children;
  }

  public final AbstractBaseGraph<V, E> getGraph()
  {
    return graph;

  }

  public String computeLexicographicEncoding()
  {
    return computeLexicographicEncoding(root);
  }

  public String computeLexicographicEncoding(V vertex)
  {
    StringBuilder encoding = new StringBuilder(vertex.toString());

    Set<E> childrenEdges = getChildEdges(vertex);
    int numChildren = childrenEdges.size();
    if (numChildren > 0)
    {
      String[] childEncodings = new String[numChildren];
      int i = 0;
      for (E childEdge : childrenEdges)
        childEncodings[i++] = ("(" + childEdge + ")")
        + computeLexicographicEncoding(graph.getEdgeTarget(childEdge));
      Arrays.sort(childEncodings);

      encoding.append("[");
      for (String childEncoding : childEncodings)
        encoding.append(childEncoding);
      encoding.append("]");
    }
    return encoding.toString();
  }

  public String computeLexicographicEncodingCached()
  {
    return computeLexicographicEncodingCached(root);
  }

  public String computeLexicographicEncodingCached(V vertex)
  {
    String cachedValue = encodingCache.get(vertex);
    if (cachedValue != null) return cachedValue;

    StringBuilder encoding = new StringBuilder(vertex.toString());

    Set<E> childrenEdges = getChildEdges(vertex);
    int numChildren = childrenEdges.size();
    if (numChildren > 0)
    {
      String[] childEncodings = new String[numChildren];
      int i = 0;
      for (E childEdge : childrenEdges)
        childEncodings[i++] = ("(" + childEdge + ")")
        + computeLexicographicEncodingCached(
            graph.getEdgeTarget(childEdge));
      Arrays.sort(childEncodings);

      encoding.append("[");
      for (String childEncoding : childEncodings)
        encoding.append(childEncoding);
      encoding.append("]");
    }
    String encodingString = encoding.toString();
    encodingCache.put(vertex, encodingString);
    return encodingString;
  }

  public void clearCache()
  {
    encodingCache.clear();
  }
}
