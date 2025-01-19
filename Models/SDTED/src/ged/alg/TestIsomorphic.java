package ged.alg;

import org.jgrapht.alg.isomorphism.VF2GraphIsomorphismInspector;
import org.jgrapht.graph.AbstractGraph;

import ged.alg.distance.comparison.ExactGraphEditDistance;
import ged.alg.graphencoder.WeisfeilerLehman;
import ged.alg.graphencoder.WeisfeilerLehmanCompressed;
import ged.alg.graphencoder.WeisfeilerLehmanCompressedLookAhead;
import ged.alg.graphencoder.tree.KNeighborhoodTree;
import ged.alg.graphencoder.tree.NeighborhoodTreeV1;
import ged.alg.graphencoder.tree.NeighborhoodTreeV1Plus;
import ged.alg.graphencoder.tree.NeighborhoodTreeV2;
import ged.alg.graphencoder.tree.NeighborhoodTreeWLLookAhead;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;

public class TestIsomorphic
{

  /**
   * @param g1
   * @param g2
   * @return Returns true iff g1 and g2 have the same encoding under V1
   */
  public static <V, E extends Edge> boolean isomorphicV1(AbstractGraph<V, E> g1,
      AbstractGraph<V, E> g2)
  {
    NeighborhoodTreeV1<V, E> nt1 = new NeighborhoodTreeV1<V, E>(g1, true);
    NeighborhoodTreeV1<V, E> nt2 = new NeighborhoodTreeV1<V, E>(g2, true);
    return nt1.computeLexicographicEncoding()
        .equals(nt2.computeLexicographicEncoding());
  }
  public static <V, E extends Edge> boolean isomorphicK(AbstractGraph<V, E> g1,
	      AbstractGraph<V, E> g2, int k)
	  {
	    KNeighborhoodTree<V, E> nt1 = new KNeighborhoodTree<V, E>(g1, true, k);
	    KNeighborhoodTree<V, E> nt2 = new KNeighborhoodTree<V, E>(g2, true, k);
	    return nt1.computeLexicographicEncoding()
	        .equals(nt2.computeLexicographicEncoding());
	  }

  public static <V, E extends Edge> boolean isomorphicV1Plus(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    NeighborhoodTreeV1Plus<V, E> nt1 = new NeighborhoodTreeV1Plus<V, E>(g1,
        true);
    NeighborhoodTreeV1Plus<V, E> nt2 = new NeighborhoodTreeV1Plus<V, E>(g2,
        true);
    return nt1.computeLexicographicEncoding()
        .equals(nt2.computeLexicographicEncoding());
  }

  public static <V, E extends Edge> boolean isomorphicV2(AbstractGraph<V, E> g1,
      AbstractGraph<V, E> g2)
  {
    NeighborhoodTreeV2<V, E> nt1 = new NeighborhoodTreeV2<V, E>(g1, true);
    NeighborhoodTreeV2<V, E> nt2 = new NeighborhoodTreeV2<V, E>(g2, true);
    return nt1.computeLexicographicEncoding()
        .equals(nt2.computeLexicographicEncoding());
  }

  public static <V, E> boolean isomorphicWLCompressed(AbstractGraph<V, E> g1,
      AbstractGraph<V, E> g2)
  {
    return new WeisfeilerLehmanCompressed<V, E>(g1, true)
        .computeLexicographicEncoding()
        .equals(new WeisfeilerLehmanCompressed<V, E>(g2, true)
            .computeLexicographicEncoding());
  }

  public static <V, E> boolean isomorphicWLCompressedLA(AbstractGraph<V, E> g1,
      AbstractGraph<V, E> g2)
  {
    return new WeisfeilerLehmanCompressedLookAhead<V, E>(g1, true)
        .computeLexicographicEncoding()
        .equals(new WeisfeilerLehmanCompressedLookAhead<V, E>(g2, true)
            .computeLexicographicEncoding());
  }

  public static <V, E extends Edge> boolean isomorphicWL(AbstractGraph<V, E> g1,
      AbstractGraph<V, E> g2)
  {
    return new WeisfeilerLehman<V, E>(g1, true).computeLexicographicEncoding()
        .equals(new WeisfeilerLehman<V, E>(g2, true)
            .computeLexicographicEncoding());
  }

  public static <V, E extends Edge> boolean isomorphicWLLA(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    return new NeighborhoodTreeWLLookAhead<V, E>(g1, true)
        .computeLexicographicEncoding()
        .equals(new NeighborhoodTreeWLLookAhead<V, E>(g2, true)
            .computeLexicographicEncoding());
  }

  public static <V, E extends Edge> boolean isomorphicIterativeV1(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    NeighborhoodTreeV1<V, E> nt1 = new NeighborhoodTreeV1<V, E>(g1);
    NeighborhoodTreeV1<V, E> nt2 = new NeighborhoodTreeV1<V, E>(g2);

    boolean refine1;
    boolean refine2;
    do
    {
      String enc1 = nt1.computeLexicographicEncoding();
      String enc2 = nt2.computeLexicographicEncoding();
      if (!enc1.equals(enc2)) return false;
      refine1 = nt1.refine();
      refine2 = nt2.refine();
    }
    while (refine1 && refine2);
    return refine1 == refine2;
  }

  public static <V, E extends Edge> boolean isomorphicIterativeV2(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    NeighborhoodTreeV2<V, E> nt1 = new NeighborhoodTreeV2<V, E>(g1);
    NeighborhoodTreeV2<V, E> nt2 = new NeighborhoodTreeV2<V, E>(g2);

    boolean refine1;
    boolean refine2;
    do
    {
      String enc1 = nt1.computeLexicographicEncoding();
      String enc2 = nt2.computeLexicographicEncoding();
      if (!enc1.equals(enc2)) return false;
      refine1 = nt1.refine();
      refine2 = nt2.refine();
    }
    while (refine1 && refine2);
    return refine1 == refine2;
  }

  /**
   * @param g1
   * @param g2
   * @return true, iff VF2 implementation of JGraphT says graphs are isomorphic,
   *         false otherwise
   */
  public static <V extends Vertex, E> boolean isomorphicExact(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    VF2GraphIsomorphismInspector<V, E> iso = //
        new VF2GraphIsomorphismInspector<V, E>(g1, g2, //
            (v1, v2) -> v1.getDistance(v2) == 0 ? 0 : 1, //
                (e1, e2) -> 0);// TODO: for edge labels use different comparator
    return iso.isomorphismExists();
  }

  /**
   * @param g1
   * @param g2
   * @return isomorphism test based on finding an edit path with cost 0, should
   *         be very slow..
   */
  public static <V extends Vertex, E extends Edge> boolean isomorphicEditPath(
      AbstractGraph<V, E> g1, AbstractGraph<V, E> g2)
  {
    return new ExactGraphEditDistance().computeGraphDistance(g1, g2) == 0;
  }

}
