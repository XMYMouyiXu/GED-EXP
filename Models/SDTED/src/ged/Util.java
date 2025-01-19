package ged;

import java.awt.Dimension;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Supplier;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.jgrapht.ext.JGraphXAdapter;
import org.jgrapht.graph.AbstractGraph;
import org.jgrapht.graph.DefaultUndirectedGraph;
import org.jgrapht.nio.Attribute;
import org.jgrapht.nio.dimacs.DIMACSImporter;

import com.mxgraph.layout.mxCircleLayout;
import com.mxgraph.layout.mxCompactTreeLayout;
import com.mxgraph.model.mxGraphModel;
import com.mxgraph.model.mxICell;
import com.mxgraph.swing.mxGraphComponent;
import com.mxgraph.util.mxConstants;
import com.mxgraph.util.mxStyleUtils;

import ged.structures.graph.LabeledEdge;
import ged.structures.graph.LabeledVertex;
import ged.structures.graph.PGraph;
import ged.structures.graph.PlaceholderEdge;
import ged.structures.graph.Vertex;

public class Util
{
  public static final Dimension DEFAULT_SIZE = new Dimension(800, 600);

  public static void appendNullUpTo(Collection<?> collection, int size)
  {
    for (int i = collection.size(); i < size; ++i)
      collection.add(null);
  }

  public static <V> int wlHash(V vertex)
  {
    return vertex.toString().hashCode();
  }

  public static int wlHash(String string) {
    return string.hashCode();
  }

  public static <V> int wlHash(V vertex, int maxDistance)
  {
    return vertex.toString().hashCode() + maxDistance;
  }

  public static <V, E> int computeMaxDist(V v, AbstractGraph<V, E> graph)
  {
    HashSet<V> visited = new HashSet<V>();
    HashSet<V> recent = new HashSet<V>();
    HashSet<V> next = new HashSet<V>();
    visited.add(v);
    recent.add(v);

    int dist = 0;
    while (!recent.isEmpty())
    {
      ++dist;
      for (V nextV : recent)
      {
        for (E e : graph.edgesOf(nextV))
        {
          V neighbor = graph.getEdgeSource(e);
          if (neighbor.equals(nextV)) neighbor = graph.getEdgeTarget(e);

          if (visited.add(neighbor)) next.add(neighbor);
        }
      }
      recent = next;
      next = new HashSet<V>();
    }
    return dist - 1;
  }

  public static <V, E> void clearEdgeLabels(JGraphXAdapter<V, E> adapter)
  {
    HashMap<E, mxICell> vertexToCell = adapter.getEdgeToCellMap();
    for (mxICell cellOfEdge : vertexToCell.values())
      adapter.cellLabelChanged(cellOfEdge, "", true);
  }

  public static <V, E> void updateVertexLabels(JGraphXAdapter<V, E> adapter,
      Map<V, Map<String, Attribute>> vertexAttributes)
  {
    for (Entry<V, mxICell> vertexWithCell : adapter.getVertexToCellMap()
        .entrySet())
    {
      Map<String, Attribute> attributesOfVertex = vertexAttributes
          .get(vertexWithCell.getKey());
      if (attributesOfVertex != null)
      {
        Attribute attribute = attributesOfVertex.get("label");
        if (attribute != null)
          adapter.cellLabelChanged(vertexWithCell.getValue(),
              attribute.getValue(), true);
      }
    }
  }

  public static <V, E> void printGraph(AbstractGraph<V, E> graph)
  {
    System.out.println(
        "|V|: " + graph.vertexSet().size() + " |E|: " + graph.edgeSet().size()
        + " 2*|E| (if undirected): " + graph.edgeSet().size() * 2);
    ArrayList vertices = new ArrayList(graph.vertexSet());
    for (int i = 0; i< vertices.size(); i++)
    {
      V v = (V) vertices.get(i);
      System.out.print(i + " " + v.toString() + ": ");
      for (E e : graph.edgesOf(v))
      {
        V u = graph.getEdgeTarget(e);
        if (u.equals(v))
        {
          u = graph.getEdgeSource(e);
        }
        System.out.print(vertices.indexOf(u) + " " + u.toString() + " (" + e.toString() + ") ");
      }
      System.out.println();
    }
  }

  public static <V, E> void showGraph(JGraphXAdapter<V, E> adapter,
      boolean directed, Dimension size)
  {
    mxGraphComponent graphComponent = new mxGraphComponent(adapter);

    if (!directed)
    {
      mxGraphModel graphModel = (mxGraphModel) graphComponent.getGraph()
          .getModel();

      // the cells to apply the style too
      Collection<Object> cells = graphModel.getCells().values();

      // make it look undirected
      mxStyleUtils.setCellStyles(graphComponent.getGraph().getModel(),
          cells.toArray(), mxConstants.STYLE_ENDARROW, mxConstants.NONE);
      // ^ careful! if you add edges later they will not have that style and
      // need to be updated
    }

    mxCircleLayout layout = new mxCircleLayout(adapter);
    int radius = 50;
    layout.setX0((size.width / 2.0) - radius);
    layout.setY0((size.height / 2.0) - radius);
    layout.setRadius(radius);
    layout.setMoveCircle(true);
    layout.execute(adapter.getDefaultParent());

    JFrame frame = new JFrame();
    frame.setSize(size);
    frame.setContentPane(graphComponent);
    frame.setVisible(true);
    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
  }

  public static <V, E> void showTreeGraph(JGraphXAdapter<V, E> adapter,
      boolean directed, Dimension size)
  {
    mxGraphComponent graphComponent = new mxGraphComponent(adapter);

    if (!directed)
    {
      mxGraphModel graphModel = (mxGraphModel) graphComponent.getGraph()
          .getModel();

      // the cells to apply the style too
      Collection<Object> cells = graphModel.getCells().values();

      // make it look undirected
      mxStyleUtils.setCellStyles(graphComponent.getGraph().getModel(),
          cells.toArray(), mxConstants.STYLE_ENDARROW, mxConstants.NONE);
      // ^ careful! if you add edges later they will not have that style and
      // need to be updated
    }

    mxCompactTreeLayout layout = new mxCompactTreeLayout(adapter);
    layout.execute(adapter.getDefaultParent());

    JFrame frame = new JFrame();
    frame.setSize(size);
    frame.setContentPane(graphComponent);
    frame.setVisible(true);
    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraph1()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("Grau"),
        new LabeledVertex("Blau"), new LabeledVertex("Rot"),
        new LabeledVertex("Gruen")};

    graph.addVertex(vertices[0]);
    graph.addVertex(vertices[1]);
    graph.addVertex(vertices[2]);
    graph.addVertex(vertices[3]);

    // do we also have to insert the backedges (it is undirected so it should
    // not matter)
    graph.addEdge(vertices[0], vertices[1]);
    graph.addEdge(vertices[0], vertices[2]);
    graph.addEdge(vertices[2], vertices[1]);
    graph.addEdge(vertices[3], vertices[1]);

    return graph;
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraph2()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("Grau"),
        new LabeledVertex("Blau"), new LabeledVertex("Rot"),
        new LabeledVertex("Gruen"), new LabeledVertex("Gelb")};

    graph.addVertex(vertices[0]);
    graph.addVertex(vertices[1]);
    graph.addVertex(vertices[2]);
    graph.addVertex(vertices[3]);
    graph.addVertex(vertices[4]);

    // do we also have to insert the backedges (it is undirected so it should
    // not matter)
    graph.addEdge(vertices[0], vertices[1]);
    graph.addEdge(vertices[0], vertices[2]);
    graph.addEdge(vertices[3], vertices[1]);
    graph.addEdge(vertices[3], vertices[2]);
    graph.addEdge(vertices[3], vertices[4]);

    return graph;
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraph3RandomizedOrder()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C"), new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C")};

    List<Supplier<?>> todo = new ArrayList<Supplier<?>>();

    todo.add(() -> graph.addVertex(vertices[0]));
    todo.add(() -> graph.addVertex(vertices[1]));
    todo.add(() -> graph.addVertex(vertices[2]));
    todo.add(() -> graph.addVertex(vertices[3]));
    todo.add(() -> graph.addVertex(vertices[4]));
    todo.add(() -> graph.addVertex(vertices[5]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();

    todo.clear();

    // do we also have to insert the backedges (it is undirected so it should
    // not matter)
    todo.add(() -> graph.addEdge(vertices[0], vertices[1]));
    todo.add(() -> graph.addEdge(vertices[0], vertices[2]));
    todo.add(() -> graph.addEdge(vertices[2], vertices[1]));
    todo.add(() -> graph.addEdge(vertices[3], vertices[1]));
    todo.add(() -> graph.addEdge(vertices[3], vertices[4]));
    todo.add(() -> graph.addEdge(vertices[4], vertices[5]));
    todo.add(() -> graph.addEdge(vertices[3], vertices[5]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();

    todo.clear();

    return graph;
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraphWLTwoRings1()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("C"), new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C"), new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C"), new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C")};

    List<Supplier<?>> todo = new ArrayList<Supplier<?>>();

    todo.add(() -> graph.addVertex(vertices[0]));
    todo.add(() -> graph.addVertex(vertices[1]));
    todo.add(() -> graph.addVertex(vertices[2]));
    todo.add(() -> graph.addVertex(vertices[3]));
    todo.add(() -> graph.addVertex(vertices[4]));
    todo.add(() -> graph.addVertex(vertices[5]));
    todo.add(() -> graph.addVertex(vertices[6]));
    todo.add(() -> graph.addVertex(vertices[7]));
    todo.add(() -> graph.addVertex(vertices[8]));
    todo.add(() -> graph.addVertex(vertices[9]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();

    todo.clear();

    // do we also have to insert the backedges (it is undirected so it should
    // not matter)
    todo.add(() -> graph.addEdge(vertices[0], vertices[1]));
    todo.add(() -> graph.addEdge(vertices[1], vertices[2]));
    todo.add(() -> graph.addEdge(vertices[2], vertices[3]));
    todo.add(() -> graph.addEdge(vertices[3], vertices[4]));
    todo.add(() -> graph.addEdge(vertices[4], vertices[5]));
    todo.add(() -> graph.addEdge(vertices[5], vertices[0]));
    todo.add(() -> graph.addEdge(vertices[5], vertices[6]));
    todo.add(() -> graph.addEdge(vertices[6], vertices[7]));
    todo.add(() -> graph.addEdge(vertices[7], vertices[8]));
    todo.add(() -> graph.addEdge(vertices[8], vertices[9]));
    todo.add(() -> graph.addEdge(vertices[9], vertices[4]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();

    todo.clear();

    return graph;
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraphWLTwoRings2()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("C"), new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C"), new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C"), new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C")};

    List<Supplier<?>> todo = new ArrayList<Supplier<?>>();

    todo.add(() -> graph.addVertex(vertices[0]));
    todo.add(() -> graph.addVertex(vertices[1]));
    todo.add(() -> graph.addVertex(vertices[2]));
    todo.add(() -> graph.addVertex(vertices[3]));
    todo.add(() -> graph.addVertex(vertices[4]));
    todo.add(() -> graph.addVertex(vertices[5]));
    todo.add(() -> graph.addVertex(vertices[6]));
    todo.add(() -> graph.addVertex(vertices[7]));
    todo.add(() -> graph.addVertex(vertices[8]));
    todo.add(() -> graph.addVertex(vertices[9]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();

    todo.clear();

    // do we also have to insert the backedges (it is undirected so it should
    // not matter)
    todo.add(() -> graph.addEdge(vertices[0], vertices[1]));
    todo.add(() -> graph.addEdge(vertices[1], vertices[2]));
    todo.add(() -> graph.addEdge(vertices[2], vertices[3]));
    todo.add(() -> graph.addEdge(vertices[3], vertices[4]));
    todo.add(() -> graph.addEdge(vertices[4], vertices[5]));
    todo.add(() -> graph.addEdge(vertices[4], vertices[0]));
    todo.add(() -> graph.addEdge(vertices[5], vertices[6]));
    todo.add(() -> graph.addEdge(vertices[6], vertices[7]));
    todo.add(() -> graph.addEdge(vertices[7], vertices[8]));
    todo.add(() -> graph.addEdge(vertices[8], vertices[9]));
    todo.add(() -> graph.addEdge(vertices[9], vertices[5]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();

    todo.clear();

    return graph;
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraphWLTrianglesRandomizedOrder()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C"), new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C")};

    List<Supplier<?>> todo = new ArrayList<Supplier<?>>();

    todo.add(() -> graph.addVertex(vertices[0]));
    todo.add(() -> graph.addVertex(vertices[1]));
    todo.add(() -> graph.addVertex(vertices[2]));
    todo.add(() -> graph.addVertex(vertices[3]));
    todo.add(() -> graph.addVertex(vertices[4]));
    todo.add(() -> graph.addVertex(vertices[5]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();

    todo.clear();

    // do we also have to insert the backedges (it is undirected so it should
    // not matter)
    todo.add(() -> graph.addEdge(vertices[0], vertices[1]));
    todo.add(() -> graph.addEdge(vertices[0], vertices[2]));
    todo.add(() -> graph.addEdge(vertices[2], vertices[1]));
    todo.add(() -> graph.addEdge(vertices[3], vertices[4]));
    todo.add(() -> graph.addEdge(vertices[4], vertices[5]));
    todo.add(() -> graph.addEdge(vertices[3], vertices[5]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();

    todo.clear();

    return graph;
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraphWLHexagonRandomizedOrder()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C"), new LabeledVertex("C"),
        new LabeledVertex("C"), new LabeledVertex("C")};

    List<Supplier<?>> todo = new ArrayList<Supplier<?>>();

    todo.add(() -> graph.addVertex(vertices[0]));
    todo.add(() -> graph.addVertex(vertices[1]));
    todo.add(() -> graph.addVertex(vertices[2]));
    todo.add(() -> graph.addVertex(vertices[3]));
    todo.add(() -> graph.addVertex(vertices[4]));
    todo.add(() -> graph.addVertex(vertices[5]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();

    todo.clear();

    // do we also have to insert the backedges (it is undirected so it should
    // not matter)
    todo.add(() -> graph.addEdge(vertices[0], vertices[1]));
    todo.add(() -> graph.addEdge(vertices[1], vertices[2]));
    todo.add(() -> graph.addEdge(vertices[2], vertices[3]));
    todo.add(() -> graph.addEdge(vertices[3], vertices[4]));
    todo.add(() -> graph.addEdge(vertices[4], vertices[5]));
    todo.add(() -> graph.addEdge(vertices[5], vertices[0]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();

    todo.clear();

    return graph;
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraph4()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("G"),
        new LabeledVertex("G"), new LabeledVertex("G"), new LabeledVertex("G"),
        new LabeledVertex("B"), new LabeledVertex("B"), new LabeledVertex("R")};

    graph.addVertex(vertices[0]);
    graph.addVertex(vertices[1]);
    graph.addVertex(vertices[2]);
    graph.addVertex(vertices[3]);
    graph.addVertex(vertices[4]);
    graph.addVertex(vertices[5]);
    graph.addVertex(vertices[6]);

    // do we also have to insert the backedges (it is undirected so it should
    // not matter)
    graph.addEdge(vertices[0], vertices[1]);
    graph.addEdge(vertices[0], vertices[4]);
    graph.addEdge(vertices[1], vertices[5]);
    graph.addEdge(vertices[4], vertices[2]);
    graph.addEdge(vertices[4], vertices[5]);
    graph.addEdge(vertices[5], vertices[3]);
    graph.addEdge(vertices[2], vertices[6]);

    return graph;
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraph5()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("G"),
        new LabeledVertex("G"), new LabeledVertex("G"), new LabeledVertex("G"),
        new LabeledVertex("B"), new LabeledVertex("B")};

    graph.addVertex(vertices[0]);
    graph.addVertex(vertices[1]);
    graph.addVertex(vertices[2]);
    graph.addVertex(vertices[3]);
    graph.addVertex(vertices[4]);
    graph.addVertex(vertices[5]);

    // do we also have to insert the backedges (it is undirected so it should
    // not matter)
    graph.addEdge(vertices[0], vertices[1]);
    graph.addEdge(vertices[0], vertices[4]);
    graph.addEdge(vertices[1], vertices[5]);
    graph.addEdge(vertices[4], vertices[2]);
    graph.addEdge(vertices[4], vertices[5]);
    graph.addEdge(vertices[5], vertices[3]);

    return graph;
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraphShrikhande()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("V"),
        new LabeledVertex("V"), new LabeledVertex("V"), new LabeledVertex("V"),
        new LabeledVertex("V"), new LabeledVertex("V"), new LabeledVertex("V"),
        new LabeledVertex("V"), new LabeledVertex("V"), new LabeledVertex("V"),
        new LabeledVertex("V"), new LabeledVertex("V"), new LabeledVertex("V"),
        new LabeledVertex("V"), new LabeledVertex("V"), new LabeledVertex("V")};
    // LabeledVertex[] vertices = new LabeledVertex[] { new LabeledVertex("0"),
    // new LabeledVertex("1"), new LabeledVertex("2"), new LabeledVertex("3"),
    // new LabeledVertex("4"), new LabeledVertex("5"), new LabeledVertex("6"),
    // new LabeledVertex("7"), new LabeledVertex("8"), new LabeledVertex("9"),
    // new LabeledVertex("10"), new LabeledVertex("11"),
    // new LabeledVertex("12"), new LabeledVertex("13"),
    // new LabeledVertex("14"), new LabeledVertex("15")};

    List<Supplier<?>> todo = new ArrayList<Supplier<?>>();

    todo.add(() -> graph.addVertex(vertices[0]));
    todo.add(() -> graph.addVertex(vertices[1]));
    todo.add(() -> graph.addVertex(vertices[2]));
    todo.add(() -> graph.addVertex(vertices[3]));
    todo.add(() -> graph.addVertex(vertices[4]));
    todo.add(() -> graph.addVertex(vertices[5]));
    todo.add(() -> graph.addVertex(vertices[6]));
    todo.add(() -> graph.addVertex(vertices[7]));
    todo.add(() -> graph.addVertex(vertices[8]));
    todo.add(() -> graph.addVertex(vertices[9]));
    todo.add(() -> graph.addVertex(vertices[10]));
    todo.add(() -> graph.addVertex(vertices[11]));
    todo.add(() -> graph.addVertex(vertices[12]));
    todo.add(() -> graph.addVertex(vertices[13]));
    todo.add(() -> graph.addVertex(vertices[14]));
    todo.add(() -> graph.addVertex(vertices[15]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();
    todo.clear();

    for (int i = 0; i < 16; ++i)
    {
      final int v = i;
      final int v2 = (v + 1) % 16;
      todo.add(() -> graph.addEdge(vertices[v], vertices[v2]));
    }

    for (int i = 0; i < 16; ++i)
    {
      final int v = i;
      final int v2 = (v + 4) % 16;
      todo.add(() -> graph.addEdge(vertices[v], vertices[v2]));
    }

    for (int i = 0; i < 15; ++i)
    {
      if (i % 4 != 3)
      {
        final int v = i;
        final int v2 = (v + 5) % 16;
        todo.add(() -> graph.addEdge(vertices[v], vertices[v2]));
      }
    }

    for (int i = 0; i < 4; ++i)
    {
      final int v = 4 * i;
      final int v2 = (v + 3);
      todo.add(() -> graph.addEdge(vertices[v], vertices[v2]));
    }

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();
    todo.clear();

    return graph;
  }

  public static DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> getExampleGraphRook4()
  {
    DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge> graph = new DefaultUndirectedGraph<LabeledVertex, PlaceholderEdge>(
        PlaceholderEdge.class);
    LabeledVertex[] vertices = { new LabeledVertex("V"),
        new LabeledVertex("V"), new LabeledVertex("V"), new LabeledVertex("V"),
        new LabeledVertex("V"), new LabeledVertex("V"), new LabeledVertex("V"),
        new LabeledVertex("V"), new LabeledVertex("V"), new LabeledVertex("V"),
        new LabeledVertex("V"), new LabeledVertex("V"), new LabeledVertex("V"),
        new LabeledVertex("V"), new LabeledVertex("V"), new LabeledVertex("V")};

    List<Supplier<?>> todo = new ArrayList<Supplier<?>>();

    todo.add(() -> graph.addVertex(vertices[0]));
    todo.add(() -> graph.addVertex(vertices[1]));
    todo.add(() -> graph.addVertex(vertices[2]));
    todo.add(() -> graph.addVertex(vertices[3]));
    todo.add(() -> graph.addVertex(vertices[4]));
    todo.add(() -> graph.addVertex(vertices[5]));
    todo.add(() -> graph.addVertex(vertices[6]));
    todo.add(() -> graph.addVertex(vertices[7]));
    todo.add(() -> graph.addVertex(vertices[8]));
    todo.add(() -> graph.addVertex(vertices[9]));
    todo.add(() -> graph.addVertex(vertices[10]));
    todo.add(() -> graph.addVertex(vertices[11]));
    todo.add(() -> graph.addVertex(vertices[12]));
    todo.add(() -> graph.addVertex(vertices[13]));
    todo.add(() -> graph.addVertex(vertices[14]));
    todo.add(() -> graph.addVertex(vertices[15]));

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();
    todo.clear();

    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        final int v = 4 * i + j;
        final int v2 = 4 * i + ((j + 1) % 4);
        todo.add(() -> graph.addEdge(vertices[v], vertices[v2]));
      }
    }

    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        final int v = 4 * i + j;
        final int v2 = 4 * ((i + 1) % 4) + j;
        todo.add(() -> graph.addEdge(vertices[v], vertices[v2]));
      }
    }

    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 2; ++j)
      {
        final int v = 4 * i + j;
        final int v2 = v + 2;
        todo.add(() -> graph.addEdge(vertices[v], vertices[v2]));
      }
    }

    for (int i = 0; i < 2; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        final int v = 4 * i + j;
        final int v2 = v + 8;
        todo.add(() -> graph.addEdge(vertices[v], vertices[v2]));
      }
    }

    Collections.shuffle(todo);
    for (Supplier<?> f : todo)
      f.get();
    todo.clear();

    return graph;
  }

  /**
   * @param path
   *          path to folder with graphs in DIMACS format
   * @return list of graphs with property "name": filename
   */
  public static ArrayList<PGraph<Vertex, PlaceholderEdge>> readDIMACSDataset(
      String path)
  {
    File folder = new File(path + "/");
    File[] listOfFiles = folder.listFiles();
    DIMACSImporter<Vertex, PlaceholderEdge> importer = new DIMACSImporter<Vertex, PlaceholderEdge>();
    importer.setVertexFactory(id -> new LabeledVertex("0"));
    // TODO: if graph has vertex labels, use them

    ArrayList<PGraph<Vertex, PlaceholderEdge>> dataset = new ArrayList<PGraph<Vertex, PlaceholderEdge>>();
    for (File f : listOfFiles)
    {
      PGraph<Vertex, PlaceholderEdge> g = new PGraph<Vertex, PlaceholderEdge>(
          PlaceholderEdge.class);
      importer.importGraph(g, f);
      g.setProperty("name", f.getName());
      dataset.add(g);
    }
    return dataset;
  }

  /**
   * @param path
   *          path to folder with dataset in TU format
   * @param name
   *          name of the dataset
   * @return list of graphs with property "class": class of graph, vertex and
   *         edge labels if present in dataset (No attributes yet!)
   * @throws IOException
   */
  public static ArrayList<PGraph<LabeledVertex, LabeledEdge>> readTUDataset(
      String path, String name) throws IOException
  {
    String prefix = path + "/" + name + "/" + name;
    ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds = new ArrayList<PGraph<LabeledVertex, LabeledEdge>>();

    // files
    File fASparse = new File(prefix + "_A.txt");
    File fNodeLabels = new File(prefix + "_node_labels.txt");
    File fNodeAttributes = new File(prefix + "_node_attributes.txt");
    File fGraphIndicator = new File(prefix + "_graph_indicator.txt");
    File fGraphLabels = new File(prefix + "_graph_labels.txt");
    File fGraphSets = new File(prefix + "_graph_sets.txt");
    File fEdgeLabels = new File(prefix + "_edge_labels.txt");
    File fEdgeAttributes = new File(prefix + "_edge_attributes.txt");

    // configuration
    boolean hasNodeLabels = fNodeLabels.exists();
    boolean hasNodeAttributes = fNodeAttributes.exists();
    boolean hasEdgeLabels = fEdgeLabels.exists();
    boolean hasEdgeAttributes = fEdgeAttributes.exists();
    boolean hasGraphSets = fGraphSets.exists();

    // open files
    BufferedReader rASparse = null;
    BufferedReader rNodeLabels = null;
    // BufferedReader rNodeAttributes = null;
    BufferedReader rGraphIndicator = null;
    BufferedReader rGraphLabels = null;
    BufferedReader rGraphSets = null;
    BufferedReader rEdgeLabels = null;
    // BufferedReader rEdgeAttributes = null;
    rASparse = new BufferedReader(new FileReader(fASparse));
    rGraphIndicator = new BufferedReader(new FileReader(fGraphIndicator));
    rGraphLabels = new BufferedReader(new FileReader(fGraphLabels));
    if (hasNodeLabels)
      rNodeLabels = new BufferedReader(new FileReader(fNodeLabels));
    if (hasNodeAttributes)
      System.err.println("Dataset has node attributes which will be ignored.");
    // rNodeAttributes = new BufferedReader(new FileReader(fNodeAttributes));
    if (hasEdgeLabels)
      rEdgeLabels = new BufferedReader(new FileReader(fEdgeLabels));
    if (hasEdgeAttributes)
      System.err.println("Dataset has edge attributes which will be ignored.");
    // rEdgeAttributes = new BufferedReader(new FileReader(fEdgeAttributes));
    if (hasGraphSets)
      rGraphSets = new BufferedReader(new FileReader(fGraphSets));

    rASparse.mark(1024);

    // read graphs
    int graphIndex = 1;
    int globalVertexIndex = 0;
    String lGraphLabel = null;
    while ((lGraphLabel = rGraphLabels.readLine()) != null)
    {
      if (lGraphLabel.equals(""))
      {
        System.out.println("Warning: Stopped reading due to line break");
        break;
      }

      // create graph
      PGraph<LabeledVertex, LabeledEdge> graph = new PGraph<LabeledVertex, LabeledEdge>(
          LabeledEdge.class);

      // set class label
      graph.setProperty("class", lGraphLabel);
      graph.setProperty("index", graphIndex);

      // set graph set
      if (hasGraphSets)
      {
        graph.setProperty("set", rGraphSets.readLine());
      }
      HashMap<Integer, LabeledVertex> idToVertex = new HashMap<Integer, LabeledVertex>();
      // create vertices
      String lGraphIndicator = null;
      while ((lGraphIndicator = rGraphIndicator.readLine()) != null)
      {
        if (Integer.valueOf(lGraphIndicator) != graphIndex)
        {
          rGraphIndicator.reset();
          break;
        }
        String lNodeLabels = hasNodeLabels ? rNodeLabels.readLine() : "";
        // String lNodeAttributes = hasNodeAttributes ?
        // rNodeAttributes.readLine() : "";
        LabeledVertex v = new LabeledVertex(lNodeLabels);
        graph.addVertex(v);
        globalVertexIndex++;
        idToVertex.put(globalVertexIndex, v);
        rGraphIndicator.mark(1024);
      }

      // create edges
      String lASparse = null;
      while ((lASparse = rASparse.readLine()) != null)
      {
        String[] tokens = lASparse.split(",");
        int iU = Integer.parseInt(tokens[0].trim());
        int iV = Integer.parseInt(tokens[1].trim());
        if (iU > globalVertexIndex || iV > globalVertexIndex)
        {
          rASparse.reset();
          break;
        }
        LabeledVertex u = idToVertex.get(iU);
        LabeledVertex v = idToVertex.get(iV);
        if (!graph.containsEdge(u, v))
        {
          String lEdgeLabels = hasEdgeLabels ? rEdgeLabels.readLine() : "";
          // String lEdgeAttributes = hasEdgeAttributes ?
          // rEdgeAttributes.readLine() : "";
          LabeledEdge e = new LabeledEdge(lEdgeLabels);
          graph.addEdge(u, v, e);
        }
        else
        {
          // skip lines
          if (hasEdgeLabels) rEdgeLabels.readLine();
          // if (hasEdgeAttributes) rEdgeAttributes.readLine();
        }
        rASparse.mark(1024);
      }

      ds.add(graph);
      graphIndex++;
    }

    // close files
    rASparse.close();
    rGraphIndicator.close();
    rGraphLabels.close();
    if (hasNodeLabels) rNodeLabels.close();
    // if (hasNodeAttributes) rNodeAttributes.close();
    if (hasEdgeLabels) rEdgeLabels.close();
    // if (hasEdgeAttributes) rEdgeAttributes.close();

    return ds;
  }

  
  public static ArrayList<PGraph<LabeledVertex, LabeledEdge>> delabelDataset(
		  ArrayList<PGraph<LabeledVertex, LabeledEdge>> oldDataset) throws IOException
	  {
	    ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds = new ArrayList<PGraph<LabeledVertex, LabeledEdge>>();
	    for(PGraph<LabeledVertex, LabeledEdge> oldGraph : oldDataset)
	    {
	      // create graph
	      PGraph<LabeledVertex, LabeledEdge> graph = new PGraph<LabeledVertex, LabeledEdge>(
	          LabeledEdge.class);

	      // set properties
	      for(String property: oldGraph.getProperties().keySet())
	      {
	    	  graph.setProperty(property, oldGraph.getProperty(property));
	      }
	      
	      
	      HashMap<LabeledVertex, LabeledVertex> oldToNewVertex = new HashMap<LabeledVertex, LabeledVertex>();
	      for(LabeledVertex v :oldGraph.vertexSet())
	      {
	    	  LabeledVertex newVertex = new LabeledVertex("");
	    	  graph.addVertex(newVertex);
	    	  oldToNewVertex.put(v, newVertex);
	      }
	       

	      // create edges
	      for(LabeledEdge edge : oldGraph.edgeSet())
	      {
	    	  LabeledEdge newEdge = new LabeledEdge("");
	    	  graph.addEdge(oldToNewVertex.get(oldGraph.getEdgeSource(edge)), oldToNewVertex.get(oldGraph.getEdgeTarget(edge)), newEdge);
	      }
	      ds.add(graph);
	    }
	    return ds;
	  }
  
  
  
  /**
   * @param path
   *          path to file with graph(s) represented as their adjacency matrix
   * @return list of graphs
   * @throws IOException
   */
  public static ArrayList<DefaultUndirectedGraph<Vertex, PlaceholderEdge>> readMatrixDataset(
      String path) throws IOException
  {
    BufferedReader reader = new BufferedReader(new FileReader(path));

    ArrayList<DefaultUndirectedGraph<Vertex, PlaceholderEdge>> dataset = new ArrayList<DefaultUndirectedGraph<Vertex, PlaceholderEdge>>();
    String line = null;
    while ((line = reader.readLine()) != null)
    {
      DefaultUndirectedGraph<Vertex, PlaceholderEdge> g = new DefaultUndirectedGraph<Vertex, PlaceholderEdge>(
          PlaceholderEdge.class);
      // add vertices
      Vertex[] vertices = new Vertex[line.length()];
      for (int i = 0; i < vertices.length; i++)
      {
        Vertex ver = new LabeledVertex("0");
        vertices[i] = ver;
        g.addVertex(ver);
      }
      int v = 0;
      while (line != null && !line.equals(""))
      {
        // add edges
        for (int u = 0; u < line.length(); u++)
        {
          if (line.charAt(u) == '1')
          {
            g.addEdge(vertices[v], vertices[u]);
          }
        }

        line = reader.readLine();
        v++;
      }
      dataset.add(g);
    }
    reader.close();
    return dataset;
  }
}
