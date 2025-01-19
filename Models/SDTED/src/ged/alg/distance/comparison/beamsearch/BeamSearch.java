package ged.alg.distance.comparison.beamsearch;
import java.util.LinkedList;
import java.util.TreeSet;

import org.jgrapht.graph.AbstractGraph;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;

/**
 * Computes the graph edit distance (exact if space in open-list is unlimited, approximated if it is not).
 * 
 * @author riesen
 * Source: https://github.com/dan-zam/graph-matching-toolkit
 *
 */
public class BeamSearch<V extends Vertex, E extends Edge> {
	private int s;
	
	public BeamSearch(int s)
	{
		this.s = s;
	}
	
	
	public double getEditDistance(AbstractGraph<V,E> g1, AbstractGraph<V,E> g2, V v1, V v2) {
		TreeNode<V,E> node = new TreeNode<V,E>(g1, g2);
		node.setMatching(v1,v2);
		return this.getEditDistance(g1, g2, node);
	}
		
	/**
	 * 
	 * @return the exact (/approximated) edit distance between graph @param g1
	 * and graph @param g2 using the cost function 
	 *
	 */
	public double getEditDistance(AbstractGraph<V,E> g1, AbstractGraph<V,E> g2, double bound) {
		if (bound <= 0.){
			return 0;
		}
		// list of partial edit paths (open) organized as TreeSet
		TreeSet<TreeNode<V,E>> open = new TreeSet<TreeNode<V,E>>();
				
		// each treenode represents a (partial) solution (i.e. edit path)
		// start is the first (empty) partial solution
		TreeNode<V,E> start = new TreeNode<V,E>(g1, g2);
		open.add(start);
		
		// time out handler
		long MAX_TIME = 30000;
		long sTime = System.currentTimeMillis();
		long eTime = sTime+1;
		
		// the successors of a node
		LinkedList<TreeNode<V,E>> successors ;
		
		// main loop of the tree search
		while (!open.isEmpty()){	
			eTime = System.currentTimeMillis();
			if (eTime-sTime > MAX_TIME){
				return -1; // max time has elapsed
			}
			TreeNode<V,E> u = open.pollFirst();
			if (u.allNodesUsed()){
				return u.getCost();
			}
			// generates all successors of node u in the search tree
			// and add them to open
			successors = u.generateSuccessors(bound);
			open.addAll(successors);
			successors.clear();
			// in beam search the maximum number of open paths
			// is limited to s
			while (open.size() > s){
				open.pollLast();
			}		
		}
		// error case 
		System.out.println("***ERROR CASE: Edit Distance is corrupt");
		System.exit(0);
		return -1;
	}
	
	/**
	 * 
	 * @return an approximate edit distance between graph @param g1
	 * and graph @param g2 using the cost function @param cf
	 * s is the maximum number of open paths used in beam-search
	 */
	public double getEditDistance(AbstractGraph<V,E> g1, AbstractGraph<V,E> g2, TreeNode<V,E> start) {
		
		// list of partial edit paths (open) organized as TreeSet
		TreeSet<TreeNode<V,E>> open = new TreeSet<TreeNode<V,E>>();
		open.add(start);
		
		// time out handler
		long MAX_TIME = 30000;
		long sTime = System.currentTimeMillis();
		long eTime = sTime+1;
		
		// the successors of a node
		LinkedList<TreeNode<V,E>> successors ;
		
		// main loop of the tree search
		while (!open.isEmpty()){	
//			System.out.println("*** OPEN: \n"+open);
			eTime = System.currentTimeMillis();
			if (eTime-sTime > MAX_TIME){
				return -1; // max time has elapsed
			}
			TreeNode<V,E> u = open.pollFirst(); 
			if (u.allNodesUsed()){
				//u.printMatching();
				return u.getCost();
			}
			// generates all successors of node u in the search tree
			// and add them to open
			successors = u.generateSuccessors(Double.MAX_VALUE);
			open.addAll(successors);
			successors.clear();
		}
		// error case 
		System.out.println("***ERROR CASE: Edit Distance is corrupt");
		System.exit(0);
		return -1;
	}
	
	


}