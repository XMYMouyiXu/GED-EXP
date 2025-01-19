package ged.alg.distance.comparison.beamsearch;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map.Entry;

import org.jgrapht.graph.AbstractGraph;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;

/**
 * Node in the search tree for the optimal edit path in order to compute the graph edit distance.
 * Each Node corresponds to a partial edit path between the two graphs induced by the matching between the vertices.
 * 
 * @author riesen
 * Source: https://github.com/dan-zam/graph-matching-toolkit
 * 
 */
public class TreeNode<V extends Vertex, E extends Edge> implements Comparable<TreeNode<V,E>> {

	/** nodes of g1 are mapped to...*/
	private int[] matching;

	/** nodes of g2 are mapped to...*/
	private int[] inverseMatching;
	
	/** the current cost of this partial solution*/
	private double cost;

	/** the original graphs */
	private AbstractGraph<V,E> originalGraph1;
	private AbstractGraph<V,E> originalGraph2;
	
	/** the graphs where the processed nodes are removed */
	private ArrayList<V> unusedVertices1;
	private ArrayList<V> unusedVertices2;
	
	/** the vertices corresponding to indices */
	private HashMap<V,Integer> vertices1;
	private HashMap<V,Integer> vertices2;
	
	private int depth;
	
	/**
	 * Constructor for the initial empty solution
	 * @param g2 graph 2
	 * @param g1 graph 1
	 */
	public TreeNode(AbstractGraph<V,E> g1, AbstractGraph<V,E> g2) {
		ArrayList<V> v1 = new ArrayList<V>();
		this.vertices1 = new HashMap<V,Integer>();
		int c=0;
		for(V v: g1.vertexSet())
		{
			v1.add(v);
			this.vertices1.put(v,c);
			c++;
		}
		c=0;
		ArrayList<V> v2 = new ArrayList<V>();
		this.vertices2 = new HashMap<V,Integer>();
		for(V v: g2.vertexSet())
		{
			v2.add(v);
			this.vertices2.put(v,c);
			c++;
		}
		
		
		this.unusedVertices1 = v1;
		this.unusedVertices2 = v2;
		this.originalGraph1 = g1;
		this.originalGraph2 = g2;
		this.cost = 0;
		this.depth = 0;
		this.matching = new int[g1.vertexSet().size()];
		this.inverseMatching = new int[g2.vertexSet().size()];
		for (int i = 0; i < this.matching.length; i++) {
			this.matching[i] = -1;
		}
		for (int i = 0; i < this.inverseMatching.length; i++) {
			this.inverseMatching[i] = -1;
		}
	}
	
	/**
	 * Copy constructor in order to generate successors 
	 * of treenode @param o
	 */
	@SuppressWarnings("unchecked")
	public TreeNode(TreeNode<V,E> o) {
		this.unusedVertices1 = (ArrayList<V>) o.getUnusedVertices1().clone();
		this.unusedVertices2 = (ArrayList<V>) o.getUnusedVertices2().clone();
		this.vertices1 = o.getVertices1();
		this.vertices2 = o.getVertices2();
		this.cost = o.getCost();
		this.matching =o.matching.clone();
		this.inverseMatching = o.inverseMatching.clone();
		this.originalGraph1 = o.originalGraph1;
		this.originalGraph2 = o.originalGraph2;
	}


	

	public HashMap<V, Integer> getVertices2() {
		return this.vertices2;
	}

	public HashMap<V, Integer> getVertices1() {
		return this.vertices1;
	}

	/**
	 * @return a list of successors of this treenode (extended solutions to 
	 * *this* solution)
	 */
	public LinkedList<TreeNode<V,E>> generateSuccessors(double bound) {
		
		bound = (double) Math.round(bound * 100000) / 100000; 
		// list with successors
		LinkedList<TreeNode<V,E>> successors = new LinkedList<TreeNode<V,E>>();
		
		// all vertices of g2 are processed, the remaining vertices of g1 are deleted
		if (this.unusedVertices2.isEmpty()) {
			TreeNode<V,E> tn = new TreeNode<V,E>(this);
			int n = tn.unusedVertices1.size();
			int e = 0;
			Iterator<V> nodeIter = tn.unusedVertices1.iterator();
			while (nodeIter.hasNext()) {
				V node = nodeIter.next();
				int i = this.vertices1.get(node);
				// find number of edges adjacent to vertex i
				e += this.getNumberOfAdjacentEdges(tn.matching,this.originalGraph1,node, this.vertices1);
				tn.matching[i] = -2; // -2 = deletion
			}
			//add costs for deleting the vertices and their adjacent edges
			tn.addCost(n * 1);  //TODO: change this, for other cost functions
			tn.addCost(e * 1);
			tn.unusedVertices1.clear();
			double c = (double)Math.round(tn.getCost() * 100000) / 100000;
			if (c <= bound){
				successors.add(tn);
			}
		} else { // there are still vertices in g2 but no vertices in g1, the vertices of g2 are inserted
			if (this.unusedVertices1.isEmpty()) {
				TreeNode<V,E> tn = new TreeNode<V,E>(this);
				int n = tn.unusedVertices2.size();
				int e = 0;
				Iterator<V> nodeIter = tn.unusedVertices2.iterator();
				while (nodeIter.hasNext()) {
					V node = nodeIter.next();
					int i = this.vertices2.get(node);
					// find number of edges adjacent to node i
					e += this.getNumberOfAdjacentEdges(tn.inverseMatching,this.originalGraph2,node, this.vertices2);
					tn.inverseMatching[i] = -2; // -2 = insertion
				}
				tn.addCost(n * 1);
				tn.addCost(e * 1);
				tn.unusedVertices2.clear();
				double c = (double)Math.round(tn.getCost() * 100000) / 100000;
				if (c <= bound){
					successors.add(tn);
				}				
			} else { // there are vertices in both g1 and g2
				for (int i = 0; i < this.unusedVertices2.size(); i++) {
					TreeNode<V,E> tn = new TreeNode<V,E>(this);
					V start = tn.unusedVertices1.remove(0);
					V end = tn.unusedVertices2.remove(i);
					tn.addCost(start.getDistance(end));  //substitution cost
					int startIndex = this.vertices1.get(start);
					int endIndex = this.vertices2.get(end);
					tn.matching[startIndex] = endIndex;
					tn.inverseMatching[endIndex] = startIndex;
					// edge processing
					this.processEdges(tn, start, end);
					double c = (double)Math.round(tn.getCost() * 100000) / 100000;
					if (c <= bound){
						successors.add(tn);
					}
				}
				// deletion of a vertex from g1 is also a valid successor
				TreeNode<V,E> tn = new TreeNode<V,E>(this);
				V deleted = tn.unusedVertices1.remove(0); //only delete the most "promising" vertex
				int i = this.vertices1.get(deleted);
				tn.matching[i] = -2; // deletion
				tn.addCost(1);
				// find number of edges adjacent to vertex i
				int e = this.getNumberOfAdjacentEdges(tn.matching, this.originalGraph1, deleted, this.vertices1);
				tn.addCost(e*1);
				double c = (double)Math.round(tn.getCost() * 100000) / 100000;
				if (c <= bound){
					successors.add(tn);
				}
			}
		}
		return successors;
	}

	/**
	 * Updates the cost of the partial edit path induced by @param tn
	 * by processing the edges of the two vertices @param start and @param end (that have not been taken into account yet).
	 * 
	 * @param tn treenode 
	 * @param start vertex of g1
	 * @param end vertex of g2
	 */
	private void processEdges(TreeNode<V,E> tn, V start, V end) {
		//process all edges that are inserted or relabeled
		for(E edge : this.originalGraph1.edgesOf(start)) // all edges adjacent to start
		{
			V u = this.originalGraph1.getEdgeSource(edge);
			if(u.equals(start))
			{
				u = this.originalGraph1.getEdgeTarget(edge);
			}
			int start2Index = this.vertices1.get(u); // neighbor of start
			if(tn.matching[start2Index]!= -1)  // other end has been handled
			{
				int end2Index = tn.matching[start2Index]; //matching of starts neighbor
				
				if (end2Index >= 0) { //vertex is not deleted
					V end2 = getVertex(this.vertices2, end2Index);
					if(this.originalGraph2.containsEdge(end, end2)) // edge is not deleted 
					{
						E edge2 = this.originalGraph2.getEdge(end, end2);
						tn.addCost(edge.getDistance(edge2));
					}
					else // edge is deleted
					{
						tn.addCost(edge.getDeletionCost());
					}
				}
				else //node is deleted, so edge has to be deleted
				{
					tn.addCost(edge.getDeletionCost());
				}
			}
		}
		
		//process all edges that might be inserted 
		for(E edge : this.originalGraph2.edgesOf(end)) // all edges adjacent to end
		{
			V u = this.originalGraph2.getEdgeSource(edge);
			if(u.equals(end))
			{
				u = this.originalGraph2.getEdgeTarget(edge);
			}
			int end2Index = this.vertices2.get(u); // neighbor of end
			if(tn.inverseMatching[end2Index]!= -1)  // other end has been handled
			{
				//start2 has to be an existing vertex,
				//since vertices are only inserted if all other vertices have been processed,
				//their edges are then processed in the same step
				
				int start2Index = tn.inverseMatching[end2Index]; //matching of ends neighbor
				V start2 = getVertex(this.vertices1, start2Index);
				if (!this.originalGraph1.containsEdge(start, start2)) // edge is inserted
				{
					tn.addCost(edge.getDeletionCost());
				}
				// costs for relabeling edges are already counted 
			}
		}
	}

	private V getVertex(HashMap<V, Integer> vertices12, int start2Index) {
		for(Entry<V, Integer> entry: vertices12.entrySet())
		{
			if(entry.getValue()== start2Index)
			{
				return entry.getKey();
			}
		}
		System.err.println("Vertex not found!");
		return null;
	}

	/**
	 * @param graph 
	 * @return number of adjacent edges of vertex with index @param i
	 * NOTE: only edges (i,j) are counted if
	 * j-th vertex has been processed (deleted or substituted)
	 */
	private int getNumberOfAdjacentEdges(int[] m, AbstractGraph<V, E> graph, V v, HashMap<V,Integer> vertices) {
		int e= 0;
		for(E edge : graph.edgesOf(v))
		{
			V u = graph.getEdgeSource(edge);
			if(u.equals(v))
			{
				u = graph.getEdgeTarget(edge);
			}
			int j = vertices.get(u);
			if (m[j]!=-1){ // count edges only if other end has been processed
				e += 1;
			}
		}
		return e;
	}

	/**
	 * adds @param c to the current solution cost
	 */
	private void addCost(double c) {
		this.cost += c;
	}
	
	/**
	 * @return true if all vertices are used in the current solution
	 */
	public boolean allNodesUsed() {
		if (unusedVertices1.isEmpty() && unusedVertices2.isEmpty()) {
			return true;
		}
		return false;
	}
	
	/**
	 * some getters and setters
	 */
	
	public ArrayList<V> getUnusedVertices1() {
		return unusedVertices1;
	}

	public ArrayList<V> getUnusedVertices2() {
		return unusedVertices2;
	}
	
	public double getCost() {
		return this.cost;
	}


	/** 
	 * In the open list treenodes are ordered according to their past cost.
	 * NOTE THAT CURRENTLY NO HEURISTIC IS IMPLEMENTED FOR ESTIMATING THE FUTURE COSTS
	 */
	@Override
	public int compareTo(TreeNode other) {
		if (this.depth-other.getDepth()==0){
			if ((this.getCost() - other.getCost())<0){
				return -1;
			} 
			if ((this.getCost() - other.getCost())>0){
				return 1;
			} 
		}
		if (this.depth - other.getDepth()<0){
			return -1;
		} 
		if (this.depth - other.getDepth()>0){
			return 1;
		}
		// we implement the open list as a TreeSet which does not allow 
		// two equal objects. That is, if two treenodes have equal cost, only one 
		// of them would be added to open, which would not be desirable
		return 1;
	}

	private int getDepth() {
		return this.depth;
	}
	
	public int[] getMatching()
	{
		return this.matching;
	}

	public void setMatching(V v1,V v2)
	{
		this.unusedVertices1.remove(v1);
		this.unusedVertices2.remove(v2);
		this.addCost(v1.getDistance(v2));  //substitution cost
		int startIndex = this.vertices1.get(v1);
		int endIndex = this.vertices2.get(v2);
		this.matching[startIndex] = endIndex;
		this.inverseMatching[endIndex] = startIndex;
		// edge processing
		this.processEdges(this, v1, v2);
	}
	

}