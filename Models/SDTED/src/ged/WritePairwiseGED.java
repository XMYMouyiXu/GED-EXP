package ged;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;

import org.jgrapht.graph.AbstractGraph;

import ged.alg.distance.GraphDistance;
import ged.structures.graph.Edge;
import ged.structures.graph.Vertex;

public class WritePairwiseGED {
	
	public static <V extends Vertex, E extends Edge> void writePairwiseGED(ArrayList<AbstractGraph<V, E>> ds, GraphDistance distance, String path, String datasetname) throws IOException
	{
		double[][] geds = computePairwiseGED(ds,distance);
		writeToFile(geds, path, datasetname,distance.getID());
	}
	public static <V extends Vertex, E extends Edge> void writePairwiseGED(ArrayList<AbstractGraph<V, E>> ds, GraphDistance distance, String datasetname) throws IOException
	{
		writePairwiseGED(ds,distance, "", datasetname);
	}
	public static <V extends Vertex, E extends Edge> double[][] computePairwiseGED(ArrayList<AbstractGraph<V, E>> ds, GraphDistance distance)
	{
		double[][] geds = new double [ds.size()][ds.size()];
		
		//the ged should be symmetric so it theoretically be enough to compute ged(i,j) and write it also to ged(j,i),
		// but especially with the approximations we use this is not always the case!
		for(int i =0; i< ds.size(); i++)
		{
			for(int j = 0; j< ds.size(); j++)
			{
				geds[i][j] = distance.computeGraphDistance(ds.get(i), ds.get(j));
			}
		}
		
		return geds;
	}
	private static void writeToFile(double[][] geds, String path, String datasetname,String distID) throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(path+"/"+datasetname+"_"+distID+"_GEDs.txt"));
		for(int i = 0; i< geds.length; i++)
		{
			for(int j =0; j<geds[i].length; j++)
			{
				writer.write(String.format(Locale.ENGLISH, "%1.2f",geds[i][j]) + "\t");
			}
			writer.newLine();
			writer.flush();
		}
		writer.close();
	}
}
