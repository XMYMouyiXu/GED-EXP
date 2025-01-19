package ged.alg.distance.comparison.walks;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.jgrapht.alg.util.Pair;
import org.jgrapht.graph.AbstractGraph;

import ged.MathUtil;
import ged.alg.distance.GraphDistance;
import ged.structures.graph.Edge;
import ged.structures.graph.LabeledEdge;
import ged.structures.graph.LabeledVertex;
import ged.structures.graph.PGraph;
import ged.structures.graph.Vertex;

public class WalkHistograms<V extends Vertex, E extends Edge> {

	private int k;
	
	private AbstractGraph<V, E> g1;
	private List<V> vertices1;
	private AbstractGraph<V, E> g2;
	private List<V> vertices2;
	
	//Product graph
	private HashMap<Pair<V,V>,V> graphsToproductgraph;
	private AbstractGraph<V, E> productgraph;
	private List<V> verticesPG;
	
	//map of vertices to histograms
	private HashMap<V, HashMap<String, Integer>> histograms;

	//matrices ^k 
	private int[][] matrix1;
	private int[][] matrix2;
	private int[][] matrixPG;
	public WalkHistograms(int k, AbstractGraph<V, E> g1, List<V> vertices1, AbstractGraph<V, E> g2,
			List<V> vertices2) {
		this.k = k;
		this.g1 = g1;
		this.vertices1 = vertices1;
		this.g2 = g2;
		this.vertices2 = vertices2;

	}
	public void init() {
		// compute the product graph, the power of adjacency matrices and the histograms
		this.matrix1 = MathUtil.matrixPower(computeMatrix(this.g1, this.vertices1), this.k);
		this.matrix2 = MathUtil.matrixPower(computeMatrix(this.g2, this.vertices2), this.k);
		
		computeProductgraph();
		this.matrixPG = MathUtil.matrixPower(computeMatrix(this.productgraph, this.verticesPG), this.k);
		this.histograms = computeHistograms();
	}

	
	private int[][] computeMatrix(AbstractGraph<V, E> g, List<V> vertices) {
		// make an adjacency matrix out of graph (indices given by indices of vertex lists);
		int[][] matrix = new int[vertices.size()][vertices.size()];
		for(int i=0; i<vertices.size();i++)
		{
			for(int j=0; j<vertices.size(); j++)
			{
				if(g.containsEdge(vertices.get(i),vertices.get(j)))
				{
					matrix[i][j] = 1;
				}
			}
		}
		return matrix;
	}
	private HashMap<V, HashMap<String, Integer>> computeHistograms() {
		// compute histograms of all vertices in g1, g2 and the product graph
		HashMap<V, HashMap<String, Integer>> map = new HashMap<V, HashMap<String, Integer>>();
		for(int i = 0; i< vertices1.size(); i++)
		{
			map.put(vertices1.get(i), computeHistogram(i, this.matrix1, this.vertices1));
		}
		for(int i = 0; i< vertices2.size(); i++)
		{
			map.put(vertices2.get(i), computeHistogram(i, this.matrix2, this.vertices2));
		}
		for(int i = 0; i< verticesPG.size(); i++)
		{
			map.put(verticesPG.get(i), computeHistogram(i, this.matrixPG, this.verticesPG));
		}
		return map;
	}
	private  HashMap<String, Integer> computeHistogram(int v, int[][] matrix, List<V> vertices) {
		HashMap<String, Integer> histogram = new HashMap<String, Integer>();
		for(int i=0; i<vertices.size();i++)
		{
			V u = vertices.get(i);
			String key = u.toString();
			if(!histogram.containsKey(key))
			{
				histogram.put(key,0);
			}
			histogram.put(key,histogram.get(key)+matrix[v][i]);// TODO is this the right direction or should it be matrix[i][v]?
		}
		return histogram;
	}
	private void computeProductgraph() {
		this.graphsToproductgraph = new HashMap<Pair<V,V>,V>();
		HashMap<V, Pair<V,V>> maptographs=new HashMap<V,Pair<V,V>>();
		this.verticesPG = new ArrayList<V>();
		this.productgraph = new PGraph<V, E>((Class<? extends E>) LabeledEdge.class);
		for(V u: this.vertices1)
		{
			for(V v: this.vertices2)
			{
				if(u.getDistance(v)==0) //they have the same label
				{
					V nV = (V) new LabeledVertex(u.toString());
					this.productgraph.addVertex(nV);
					this.verticesPG.add(nV);
					this.graphsToproductgraph.put(new Pair<V,V>(u,v),nV); //note correspondence of vertex to vertices
					maptographs.put(nV, new Pair<V,V>(u,v)); //is this needed anywhere else?
				}
			}
		}
		for(V u: this.verticesPG)
		{
			for(V v: this.verticesPG)
			{
				if(!this.productgraph.containsEdge(u,v))
				{
					Pair<V, V> pair1 = maptographs.get(u);
					Pair<V, V> pair2 = maptographs.get(v);
					if(this.g1.containsEdge(pair1.getFirst(),pair2.getFirst()) && this.g2.containsEdge(pair1.getSecond(),pair2.getSecond()) )
					{
						if(this.g1.getEdge(pair1.getFirst(),pair2.getFirst()).getDistance(this.g2.getEdge(pair1.getSecond(),pair2.getSecond()))==0)
						{
							this.productgraph.addEdge(u,v);
						}
					}
				}
			}
		}
		
	}
	public double vertexSubstitution(V v1, V v2) {
		//compute the costs for substitution according to definition
		double cost = 0;
		double delta = 0;
		if(v1.getDistance(v2)!=0) // starting at vertices with different labels
		{
			delta = 1;
		}
		
		//different paths ending with same label
		HashMap<String, Integer> h_ij =new HashMap<String, Integer>();
		HashMap<String, Integer> h_ji =new HashMap<String, Integer>();
		HashMap<String, Integer> h_i = this.histograms.get(v1);
		HashMap<String, Integer> h_j = this.histograms.get(v2);
		for(String key: h_i.keySet())
		{
			h_ij.put(key, h_i.get(key));
			h_ji.put(key, 0);
		}
		for(String key: h_j.keySet())
		{
			h_ji.put(key, h_j.get(key));
			if(!h_ij.containsKey(key))
			{
				h_ij.put(key, 0);
			}
		}
		//reduce by number of shared paths in PG
		if(this.graphsToproductgraph.containsKey(new Pair(v1,v2)))
		{
			V v_ij = this.graphsToproductgraph.get(new Pair(v1,v2));
			HashMap<String, Integer> h_x = this.histograms.get(v_ij);
			for(String key: h_i.keySet())
			{
				if(h_x.containsKey(key)) //otherwise min is 0 anyways
				{
					h_ij.put(key, h_i.get(key)-Math.min(Math.min(h_i.get(key),h_j.get(key)),(int) Math.floor(Math.sqrt(h_x.get(key)))));
				}
			}
			for(String key: h_j.keySet())
			{
				if(h_x.containsKey(key))
				{
					h_ji.put(key, h_j.get(key)-Math.min(Math.min(h_i.get(key),h_j.get(key)),(int) Math.floor(Math.sqrt(h_x.get(key)))));
				}
			}
		}
		// count walks not similar, where end nodes also have to be substituted
		int r_ij =0;
		int r_ji =0;
		for(String key: h_ij.keySet())
		{
			r_ij+= h_ij.get(key)- Math.min(h_ij.get(key),h_ji.get(key));
		}
		for(String key: h_ji.keySet())
		{
			r_ji+= h_ji.get(key)- Math.min(h_ij.get(key),h_ji.get(key));
		}
		
		
		// not similar walks ending with nodes with same label
		double sum = computeSumOverMinimums(v1, v2, h_ij, h_ji);
		cost += ((delta + this.k -1)*GraphDistance.VERTEX_SUBSTITUTION_COSTS+ this.k*GraphDistance.EDGE_SUBSTITUTION_COSTS)*sum;
		
		// not similar walks ending with different labels
		cost += ((delta + this.k)*GraphDistance.VERTEX_SUBSTITUTION_COSTS+ this.k*GraphDistance.EDGE_SUBSTITUTION_COSTS)*Math.min(r_ij, r_ji);
		
		// walks that have to be removed/inserted
		cost += ((delta + this.k)*GraphDistance.VERTEX_INSERTION_DELETION_COSTS+ this.k*GraphDistance.EDGE_INSERTION_DELETION_COSTS)*Math.abs(r_ij-r_ji);
		
		
		return cost;
	}
	
	private double computeSumOverMinimums(V v1, V v2, HashMap<String, Integer> h_ij, HashMap<String, Integer> h_ji) {
		int r = 0;
		for(String key: h_ij.keySet()) //this should contain all keys in h_ij, h_ji
		{
			r+= Math.min(h_ij.get(key),h_ji.get(key));
		}
		return r;
	}
	public double vertexInsertionDeletion(V v) {
		// cost = ((k+1)*c_delnode + k*c_deledge*|B_v|
		return ((this.k+1)*GraphDistance.VERTEX_INSERTION_DELETION_COSTS + this.k*GraphDistance.EDGE_INSERTION_DELETION_COSTS)*bucketsize(v);
	}
	
	private int bucketsize(V v) {
		// add up all entries in histogram
		HashMap<String, Integer> histogram = this.histograms.get(v);
		int count = 0;
		for(String key : histogram.keySet())
		{
			count+=histogram.get(key);
		}
		return count;
	}
}
