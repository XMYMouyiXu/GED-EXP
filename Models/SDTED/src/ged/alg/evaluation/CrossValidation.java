package ged.alg.evaluation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import ged.structures.graph.Edge;
import ged.structures.graph.PGraph;
import ged.structures.graph.Vertex;

public class CrossValidation<V extends Vertex,E extends Edge>  {

	public static final int FOLDS = 10;
	public static final int REPETITIONS = 10;

	private Random rng;
	private ArrayList<PGraph<V,E>>   dataset;
	private KNNClassification<V,E> classifier;

	public CrossValidation(ArrayList<PGraph<V,E>> dataset, KNNClassification<V,E> classifier) {
		this.dataset = dataset;
		this.classifier = classifier;
	}

	public void run() {

		double[] accuracy = new double[REPETITIONS];
		rng = new Random(42); // same partitions for each call
		for (int i = 0; i < REPETITIONS; i++) {
			//System.out.println();
			//System.out.println("REPETITION  " + (i + 1) + "  of  " + REPETITIONS);
			ArrayList<ArrayList<PGraph<V,E>>> folds = partition(dataset, FOLDS);
			double acc = crossValidate(folds);
			accuracy[i] = acc;
			//System.out.println("Avg. Accuracy: " + acc);
		}
		System.out.println("========================");
		double avg = 0;
		for (double d : accuracy)
			avg += d;
		avg /= REPETITIONS;

		// compute stdev
		double stdev = 0;
		for (double d : accuracy)
			stdev += (d - avg) * (d - avg);
		stdev = Math.sqrt(stdev / REPETITIONS);

		System.out.println("Accuracy: \t\t\t\t" + avg);
		System.out.println("Standard deviation: \t" + stdev);
	}

	public double crossValidate(ArrayList<ArrayList<PGraph<V,E>>> folds) {
		double avgAccuracy = 0;
		for (int iFold = 0; iFold < folds.size(); iFold++) {

			//System.out.println("\tFold  " + (iFold + 1) + "  of  " + folds.size());

			ArrayList<PGraph<V,E>> testSet = new ArrayList<PGraph<V,E>>();
			testSet.addAll(folds.get(iFold));
			ArrayList<PGraph<V,E>> trainingSet = new ArrayList<PGraph<V,E>>();
			for (int i = 0; i < folds.size(); i++) {
				if (i != iFold) {
					trainingSet.addAll(folds.get(i));
				}
			}

			// training
			classifier.train(trainingSet);

			// prediction
			double accuracy = accuracy(classifier, testSet);
			//System.out.println("\tReached Accuracy: " + accuracy);
			avgAccuracy += accuracy / folds.size();

		}

		return avgAccuracy;
	}

	/**
	 * Creates a random partition of the given data set.
	 * 
	 * @param dataset the data set
	 * @param cells   number of cells of the desired partition
	 * @return the partition
	 */
	private ArrayList<ArrayList<PGraph<V,E>>> partition(Collection<PGraph<V,E>> dataset, int cells) {
		ArrayList<ArrayList<PGraph<V,E>>> r = new ArrayList<>(cells);
		for (int i = 0; i < cells; i++) {
			r.add(new ArrayList<PGraph<V,E>>());
		}

		ArrayList<PGraph<V,E>> pool = new ArrayList<PGraph<V,E>>(dataset);

		int i = 0;
		while (!pool.isEmpty()) {
			int index = rng.nextInt(pool.size());
			PGraph<V,E> e = pool.get(index);
			int lastIndex = pool.size() - 1;
	        pool.set(index, pool.get(lastIndex));
	        pool.remove(lastIndex);
			r.get(i).add(e);
			i = ++i % cells;
		}

		return r;
	}
	
	public  double accuracy( KNNClassification<V,E> classifier, ArrayList<PGraph<V,E>> testSet) {
		
		// prediction
		int correct = 0;
		for (PGraph<V,E> g : testSet) {
			String predictedClass = classifier.predict(g);
			if (predictedClass.equals(g.getProperty("class")))correct++;
		}

		double accuracy = (double)correct/testSet.size();
		return accuracy;
	}

}
