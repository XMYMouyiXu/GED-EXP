package ged.alg.graphencoder;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.jgrapht.graph.AbstractGraph;

public class WeisfeilerLehman<V, E> extends GraphEncoder<V, E>
{
  private final LinkedHashMap<V, String> verticesToHash;
  private final LinkedHashMap<String, List<V>> hashToVertices;
  private boolean refined = false;

  public WeisfeilerLehman(AbstractGraph<V, E> graph)
  {
    super(graph);
    List<V> vertices = new ArrayList<V>(graph.vertexSet());
    int numVertices = vertices.size();
    verticesToHash = new LinkedHashMap<V, String>(numVertices);
    hashToVertices = new LinkedHashMap<String, List<V>>(numVertices);

    vertices.sort((v1, v2) -> v1.toString().compareTo(v2.toString()));

    if (numVertices == 0) return;

    String previous = null;
    List<V> verticesWithHash = null;
    for (V v : vertices)
    {
      String enc = v.toString();
      if (enc != previous) // TODO: why not check if enc is in hashTovertices
        // instead of sorting?
      {
        hashToVertices.put(enc, verticesWithHash = new ArrayList<V>());
        previous = enc;
      }
      verticesToHash.put(v, enc);
      verticesWithHash.add(v);
    }

    if (hashToVertices.size() == numVertices) refined = true;
  }

  public WeisfeilerLehman(AbstractGraph<V, E> graph, boolean refine)
  {
    this(graph);
    if (refine) fullyRefine();
  }

  @Override
  public boolean refine()
  {
    if (refined) return false;

    Map<String, List<V>> encodingMap = new HashMap<String, List<V>>();
    for (Entry<V, String> entry : verticesToHash.entrySet())
    {
      V v = entry.getKey();
      List<String> neighborHashes = new ArrayList<String>();
      for (E e : graph.edgesOf(v))
      {
        V neighbor = graph.getEdgeSource(e);
        if (neighbor.equals(v)) neighbor = graph.getEdgeTarget(e);
        neighborHashes.add(verticesToHash.get(neighbor));
      }

      Collections.sort(neighborHashes);
      String encodingString = entry.getValue() + neighborHashes.toString();

      List<V> verticesWithEncoding = encodingMap.get(encodingString);
      if (verticesWithEncoding == null) encodingMap.put(encodingString,
          verticesWithEncoding = new ArrayList<V>());
      verticesWithEncoding.add(v);
    }

    if (!checkChanged(encodingMap.values()))
    {
      refined = true;
      return false;
    }

    List<Entry<String, List<V>>> entryList = new ArrayList<>(
        encodingMap.entrySet());
    entryList.sort((e1, e2) -> e1.getKey().compareTo(e2.getKey()));

    verticesToHash.clear();
    hashToVertices.clear();

    for (Entry<String, List<V>> entry : entryList)
    {
      String hash = entry.getKey();
      List<V> verticesWithHash = entry.getValue();
      hashToVertices.put(hash, verticesWithHash);
      for (V vertexWithHash : verticesWithHash)
        verticesToHash.put(vertexWithHash, hash);
    }

    if (hashToVertices.size() == verticesToHash.size())
    {
      refined = true;
      return false;
    }
    return true;
  }

  @Override
  public String computeLexicographicEncoding()
  {

    StringBuilder encoding = new StringBuilder();
    for (Entry<String, List<V>> entry : hashToVertices.entrySet())
      encoding.append(entry.getKey() + ":" + entry.getValue().size() + " ");
    return encoding.toString();
  }

  private boolean checkChanged(Collection<List<V>> groups)
  {
    for (List<V> vertexGroup : groups)
    {
      int groupSize = vertexGroup.size();
      if (groupSize == 0) continue;

      String hash = verticesToHash.get(vertexGroup.get(0));
      for (int i = 1; i < groupSize; ++i)
        if (!hash.equals(verticesToHash.get(vertexGroup.get(i))))//
          return true;
      if (hashToVertices.get(hash).size() != groupSize)//
        return true;
    }
    return false;
  }

}
