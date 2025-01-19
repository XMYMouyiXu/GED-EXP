package ged.structures.graph;

import java.util.HashMap;

import org.jgrapht.graph.DefaultUndirectedGraph;

public class PGraph<V, E> extends DefaultUndirectedGraph<V, E>
{
  private static final long serialVersionUID = -6459168399010276744L;

  // graph with properties
  private final HashMap<String, Object> properties;

  public PGraph(Class<? extends E> edgeClass)
  {
    super(edgeClass);
    this.properties = new HashMap<String, Object>();
  }

  public HashMap<String, Object> getProperties()
  {
    return properties;
  }

  public void setProperty(String property, Object value)
  {
    this.properties.put(property, value);
  }

  public Object getProperty(String property)
  {
    return this.properties.get(property);
  }
}
