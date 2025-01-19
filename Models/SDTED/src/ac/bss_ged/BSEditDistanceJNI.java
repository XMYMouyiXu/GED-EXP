package ac.bss_ged;

public class BSEditDistanceJNI {

    static {
        System.loadLibrary("bss_ged_jni");
    }

    /**
     * queryVertices and graphVertices are properties of the vertices.
     * Each vertex has one property; therefore, the arrays' length is equal to the number of vertices.
     *
     * queryEdges and graphEdges are the edges. The dimension of them is the number of edges x 3.
     * The first and the second elements are the indexes of the adjacent vertices, and the third is the edge's property.
     */
    public native int getEditDistance(int width,
                                      int[] queryVertices, int[][] queryEdges,
                                      int[] graphVertices, int[][] graphEdges,
                                      int bound);
}