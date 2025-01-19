package ged.alg.evaluation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import ged.alg.distance.GraphDistance;
import ged.structures.graph.Edge;
import ged.structures.graph.PGraph;
import ged.structures.graph.Vertex;

public class KNNClassification<V extends Vertex,E extends Edge> {
	
	ArrayList<PGraph<V,E>>  trainingSet;
	int k;
	GraphDistance d;
	
	public KNNClassification(int k, GraphDistance d) {
		this.k = k;
		this.d = d;
	}

	public void train(ArrayList<PGraph<V,E>> data) {
		trainingSet = data;		
	}

	public String predict(PGraph<V,E> e) {
		ArrayList<PGraph<V,E>> nn = new ArrayList<>(k);
		double[] nnDist = new double[k];
		
		// populate with first k entries
		double maxDist = Double.NEGATIVE_INFINITY;
		int maxDistIndex = -1;
		for (int i=0; i<k; i++) {
			PGraph<V,E> t = trainingSet.get(i);
			double dist = d.computeGraphDistance(t, e);
			nn.add(t);
			nnDist[i] = dist;
			if (dist > maxDist) {
				maxDist = dist;
				maxDistIndex = i;
			}
		}
		
		// iterate through other entries
		for (int i=k; i<trainingSet.size(); i++) {
			PGraph<V,E>  t = trainingSet.get(i);
			double dist = d.computeGraphDistance(t, e);

			if (dist < maxDist) {
				nn.set(maxDistIndex, t);
				nnDist[maxDistIndex] = dist;
				// find new maximum distance
				maxDist = nnDist[0];
				maxDistIndex = 0;
				for (int j=1; j<k; j++) {
					if (nnDist[j] > maxDist) {
						maxDist = nnDist[j]; 
						maxDistIndex = j;
					}
				}
			}
		}
		
		// find majority class label of nns.
		HashMap<String, Integer> nnClassLabels = new HashMap<String, Integer>();
		for (int i=0; i<k; i++) {
			if(!nnClassLabels.containsKey(nn.get(i).getProperty("class").toString()))
			{
				nnClassLabels.put(nn.get(i).getProperty("class").toString(),0);
			}
			nnClassLabels.put(nn.get(i).getProperty("class").toString(),nnClassLabels.get(nn.get(i).getProperty("class").toString())+1);
		}
		double max = Double.NEGATIVE_INFINITY;
		String maxLabel = null;
		for (Entry<String, Integer> classLabel : nnClassLabels.entrySet()) {
			if (classLabel.getValue() > max) {
				maxLabel = classLabel.getKey();
			}
		}
		
		return maxLabel;
	}



}
