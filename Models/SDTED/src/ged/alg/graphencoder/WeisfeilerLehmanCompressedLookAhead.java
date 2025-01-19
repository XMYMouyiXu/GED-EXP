package ged.alg.graphencoder;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.jgrapht.graph.AbstractGraph;

import ged.Util;

public class WeisfeilerLehmanCompressedLookAhead<V, E>
extends
GraphEncoder<V, E>
{
  private static class VertexWithMaxDist<V>
  {
    V vertex;
    int maxDist;
    public VertexWithMaxDist(V vertex, int maxDist)
    {
      this.vertex = vertex;
      this.maxDist = maxDist;
    }

    @Override
    public int hashCode()
    {
      return Util.wlHash(vertex, maxDist);
    }
  }
  private final Comparator<VertexWithMaxDist<V>> comp = (o1, o2) -> Integer
      .compare(o1.hashCode(), o2.hashCode());

  private final LinkedHashMap<V, Integer> verticesToHash;
  private final LinkedHashMap<Integer, List<V>> hashToVertices;
  private boolean refined = false;

  public WeisfeilerLehmanCompressedLookAhead(AbstractGraph<V, E> graph)
  {
    super(graph);
    List<VertexWithMaxDist<V>> vertices = //
        new ArrayList<VertexWithMaxDist<V>>(graph.vertexSet().size());
    for (V vertex : graph.vertexSet())
      vertices.add(new VertexWithMaxDist<V>(//
          vertex, Util.computeMaxDist(vertex, graph)));

    int numVertices = vertices.size();
    verticesToHash = new LinkedHashMap<V, Integer>(numVertices);
    hashToVertices = new LinkedHashMap<Integer, List<V>>(numVertices);

    if (numVertices == 0) return;
    vertices.sort(comp);

    for (VertexWithMaxDist<V> ver : vertices)
    {
      V v = ver.vertex;
      int enc = ver.hashCode();
      List<V> vertWithEnc = hashToVertices.get(enc);
      if (vertWithEnc == null)
        hashToVertices.put(enc, vertWithEnc = new ArrayList<V>());
      verticesToHash.put(v, enc);
      vertWithEnc.add(v);
    }

    if (hashToVertices.size() == numVertices) refined = true;
  }

  public WeisfeilerLehmanCompressedLookAhead(AbstractGraph<V, E> graph,
      boolean refine)
  {
    this(graph);
    if (refine) fullyRefine();
  }

  @Override
  public boolean refine()
  {
    if (refined) return false;

    Map<String, List<V>> encodingMap = new HashMap<String, List<V>>();
    for (Entry<V, Integer> entry : verticesToHash.entrySet())
    {
      V v = entry.getKey();
      List<Integer> neighborHashes = new ArrayList<Integer>();
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
      int enc = Util.wlHash(entry.getKey());
      List<V> verticesWithHash = entry.getValue();
      hashToVertices.put(enc, verticesWithHash);
      for (V vertexWithHash : verticesWithHash)
        verticesToHash.put(vertexWithHash, enc);
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
    for (Entry<Integer, List<V>> entry : hashToVertices.entrySet())
      encoding.append(entry.getKey() + ":" + entry.getValue().size() + " ");
    // TODO: The encoding does not really reflect the structure of the graph.
    // Only the original vertex label of the vertices is used for the
    // encoding.
    // TODO: If we compared the encodings of two graph, they might be the
    // same.
    // Either we have to refine the two graphs simultaneously or use the
    // labels
    // of the neighbors (and their neighbors and so on) in the encoding.

    return encoding.toString();
  }

  private boolean checkChanged(Collection<List<V>> groups)
  {
    for (List<V> vertexGroup : groups)
    {
      int groupSize = vertexGroup.size();
      if (groupSize == 0) continue;

      Integer hash = verticesToHash.get(vertexGroup.get(0));
      for (int i = 1; i < groupSize; ++i)
        if (!hash.equals(verticesToHash.get(vertexGroup.get(i))))//
          return true;
      if (hashToVertices.get(hash).size() != groupSize)//
        return true;
    }
    return false;
  }

}
