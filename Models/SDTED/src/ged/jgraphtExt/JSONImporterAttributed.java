package ged.jgraphtExt;

import java.util.HashMap;
import java.util.Map;

import org.jgrapht.nio.Attribute;
import org.jgrapht.nio.json.JSONImporter;

public class JSONImporterAttributed<V, E> extends JSONImporter<V, E>
{
  private final Map<V, Map<String, Attribute>> vertexAttributes = new HashMap<>();

  private final Map<E, Map<String, Attribute>> edgeAttributes = new HashMap<>();


  public JSONImporterAttributed()
  {
    super();
    addVertexAttributeConsumer((p, a) -> {
      Map<String, Attribute> attrs = vertexAttributes.get(p.getFirst());
      if (attrs == null)
      {
        attrs = new HashMap<>();
        vertexAttributes.put(p.getFirst(), attrs);
      }
      attrs.put(p.getSecond(), a);
    });

    addEdgeAttributeConsumer((p, a) -> {
      Map<String, Attribute> attrs = edgeAttributes.get(p.getFirst());
      if (attrs == null)
      {
        attrs = new HashMap<>();
        edgeAttributes.put(p.getFirst(), attrs);
      }
      attrs.put(p.getSecond(), a);
    });
  }

  public Map<V, Map<String, Attribute>> getVertexAttributes()
  {
    return vertexAttributes;
  }

  public Map<E, Map<String, Attribute>> getEdgeAttributes()
  {
    return edgeAttributes;
  }
}
