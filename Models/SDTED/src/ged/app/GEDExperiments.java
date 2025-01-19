package ged.app;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Random;

import org.jgrapht.alg.util.Pair;

import ged.Util;
import ged.alg.distance.GraphDistance;
import ged.alg.distance.SDTEDbasedDistanceV1;
import ged.alg.distance.SDTEDbasedDistanceV2;
import ged.alg.distance.SDTEDbasedDistanceWL;
import ged.alg.distance.comparison.BSSExactGraphEditDistance;
import ged.alg.distance.comparison.BipartiteGraphMatching;
import ged.alg.distance.comparison.SubgraphMatching;
import ged.alg.distance.comparison.WalksMatching;
import ged.structures.graph.LabeledEdge;
import ged.structures.graph.LabeledVertex;
import ged.structures.graph.PGraph;

public class GEDExperiments {
	static String path = "my_results";
	static String dspath = "data/TUDatasets";
	static String[] allDatasets = {
									"AIDS700nef", 
									"LINUX", 
									"IMDBMulti"
									}; 
	//, "PTC_FM","Letter-med", "proteincom_sampled","MSRC_9","NCI1"
	static String[] datasets = {
								"AIDS700nef", 
								"LINUX", 
								"IMDBMulti"
								}; 
								//, "PTC_FM", "Letter-med", "proteincom_sampled"};
	static String [] dss = {};//"MSRC_9","NCI1"
  public static void main(String[] args)
      throws IOException, InterruptedException
  {
	//   // run all experiments
	//   allExperiments();
	//   // create scatter plots
	//   scatterPlots();
	myExperiments();
  }

  public static void myExperiments()  throws IOException, InterruptedException
  {
	// dspath, 
	// path, 
	// ds, 
	// dists, 
	// names

	for (String dsName : datasets) {
		GraphDistance distance = new SDTEDbasedDistanceV2(10, 0.5);
		ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds = Util.readTUDataset(dspath, dsName);
		ArrayList<Pair<Integer, Integer>> pairs = readPairs(path, dsName);
		BufferedWriter rwriter = new BufferedWriter(
			new FileWriter(path + "/" + dsName + ".txt"));
		BufferedWriter twriter = new BufferedWriter(
			new FileWriter(path + "/" + dsName + "_time.txt"));
		for (Pair<Integer, Integer> pair : pairs) {
			long timeStart = System.nanoTime();
			double dist = distance.computeGraphDistance(ds.get(pair.getFirst()), ds.get(pair.getSecond()));
			long timeEnd = System.nanoTime();
			long runtime = timeEnd - timeStart;
			rwriter.write(pair.getFirst() + " " + pair.getSecond() + " " + dist);
			rwriter.newLine();
			rwriter.flush();

			twriter.write(pair.getFirst() + " " + pair.getSecond() + " " + runtime);
			twriter.newLine();
			twriter.flush();
		}
		twriter.close();
		rwriter.close();
	}

  }

  public static void scatterPlots() throws IOException, InterruptedException
  {
	  ArrayList<String> names = new ArrayList<String>();
	  String dist = new SDTEDbasedDistanceV1(10, 0.5).getID();
	  names.add(new BipartiteGraphMatching().getID());
	  names.add(new SubgraphMatching(2).getID());
	  names.add(new WalksMatching(6).getID());
	  for(String ds: allDatasets)
	  {
	    System.out.println(ds);
	    writeResultsScatterPlot(ds,path, dist, names);
	   }
  }
  
  public static void allExperiments()  throws IOException, InterruptedException
  {
	    ArrayList<GraphDistance> dists = new ArrayList<GraphDistance>();
	    ArrayList<String> names = new ArrayList<String>();

	    GraphDistance d = new BipartiteGraphMatching();
	    dists.add(d);
	    names.add(d.getID());
	    
	    for (int i = 1; i <= 10; i++)
	    {
	        d=new SDTEDbasedDistanceV1(i, 0.5);
	        dists.add(d);
	        names.add(d.getID());
	    }
	    for (int i = 1; i <= 10; i++)
	    {
	    	d=new SDTEDbasedDistanceV2(i, 0.5);
	        dists.add(d);
	        names.add(d.getID());
	    }
	    for (int i = 1; i <= 10; i++)
	    {
	    	d=new SDTEDbasedDistanceWL(i, 0.5);
	        dists.add(d);
	        names.add(d.getID());
	    }
	    
	    for (int i = 1; i <= 10; i++)
	    {	
	    	d = new WalksMatching(i);
	    	dists.add(d);
	    	names.add(d.getID());
	    }
	    
	    for (int i = 1; i <= 2; i++) //TODO: implement time out?
	    {
	    	 d=new SubgraphMatching(i);
	    	 dists.add(d);
	    	 names.add(d.getID());
	    }
	    
	    d=new BSSExactGraphEditDistance(50);
	    dists.add(d);
	    names.add(d.getID());

		boolean generateNewPairs=false;
	    if(generateNewPairs)
	    {
			int n = 100;
			int seed = 42;

			for (String dsName : datasets) {
				ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds = Util.readTUDataset(dspath, dsName);
				generateAndWritePairs(ds.size(), n, seed, path, dsName);
			}
		}
	    
	    // approximation experiments
	    for(String ds: datasets)
	    {
	    	System.out.println(ds);
	    	approximationExperiment(dspath, path, ds, dists, names);
	    	writeResultsApproximationError(ds, path, d.getID(), names);
	    }
	    
	    //runtime experiments
	    System.out.println("Runtime");
	    for(String ds: datasets)
	    {
	    	System.out.println(ds);
	    	runtimeExperiment(dspath, path, ds, dists, names);
	    }
	    
	    //larger datasets: remove exact distance
	    dists.remove(d);
	    names.remove(d.getID());
	    // approximation experiments
	    for(String ds: dss)
	    {
	    	System.out.println(ds);
	    	approximationExperiment(dspath, path, ds, dists, names);
	    	writeResultsApproximationError(ds, path, new BipartiteGraphMatching().getID(), names);
	    }
	    
	    //runtime experiments
	    System.out.println("Runtime");
	    for(String ds: dss)
	    {
	    	System.out.println(ds);
	    	runtimeExperiment(dspath, path, ds, dists, names);
	    }
  }
  
  public static void approximationExperiment(String dsPath, String path, String dsName, ArrayList<GraphDistance> dists, ArrayList<String> distanceNames) throws IOException
  {
	  ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds = Util.readTUDataset(dsPath, dsName);
	  ArrayList<Pair<Integer, Integer>> pairs = readPairs(path, dsName);
	  for(int i = 0; i<distanceNames.size(); i++)
	  {
		  computeAndWriteDistances(path, dsName, dists.get(i), ds, pairs);
	  }
  }
  public static void runtimeExperiment(String dsPath, String path, String dsName, ArrayList<GraphDistance> dists, ArrayList<String> distanceNames) throws IOException, InterruptedException
  {
	  ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds = Util.readTUDataset(dsPath, dsName);
	  ArrayList<Pair<Integer, Integer>> pairs = readPairs(path, dsName);
	  while(pairs.size()>20) //otherwise it takes too much time
	  {
		  pairs.remove(0);
	  }
	  evaluateRuntime(dists,path, dsName,  ds, pairs);
  }
  
  public static void computeAndWriteDistances(String path, String dsname, GraphDistance distance,
			ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds, ArrayList<Pair<Integer, Integer>> pairs)
			throws IOException {
		BufferedWriter rwriter = new BufferedWriter(
				new FileWriter(path + "/" + dsname + "_" + distance.getID() + ".txt"));
		for (Pair<Integer, Integer> pair : pairs) {
			double dist = distance.computeGraphDistance(ds.get(pair.getFirst()), ds.get(pair.getSecond()));
			rwriter.write(pair.getFirst() + " " + pair.getSecond() + " " + dist);
			rwriter.newLine();
			rwriter.flush();
		}
		rwriter.close();
  }
  
  public static ArrayList<Double> readResults(String path, String dsname, String distanceName) throws NumberFormatException, IOException
  {
	  ArrayList<Double> results = new ArrayList<Double>();
	  BufferedReader reader = new BufferedReader(
  	        new FileReader(path + "/" + dsname + "_" + distanceName + ".txt"));
      String line = null;
      while((line= reader.readLine())!=null)
      {
    	  String[] tokens = line.split(" ");
        results.add(Double.parseDouble(tokens[2]));
      }
      reader.close();
      return results;
  }
  
  public static void generateAndWritePairs(int maxn, int n, int seed, String path, String dsname) throws IOException
  {
	  ArrayList<Pair<Integer, Integer>> rands = new ArrayList<Pair<Integer, Integer>>();
      Random r = new Random(seed);
      for (int i = 0; i < n; i++)
      {
        rands.add(new Pair<Integer, Integer>(r.nextInt(maxn),
            r.nextInt(maxn)));
      }
      BufferedWriter rwriter = new BufferedWriter(
    	        new FileWriter(path + "/"+ dsname+"_pairs"+".txt"));
      for(Pair<Integer, Integer> pair: rands)
      {
    	  rwriter.write(pair.getFirst()+ " "+ pair.getSecond());
    	  rwriter.newLine();
      }
      
      rwriter.close();
	  
  }
  
  public static ArrayList<Pair<Integer, Integer>> readPairs(String path, String dsname) throws IOException
  {
	  ArrayList<Pair<Integer, Integer>> rands = new ArrayList<Pair<Integer, Integer>>();
	  BufferedReader reader = new BufferedReader(
  	        new FileReader(path + "/"+ dsname+"_pairs"+".txt"));
      String line = null;
      while((line= reader.readLine())!=null)
      {
    	  String[] tokens = line.split(" ");
        rands.add(new Pair<Integer, Integer>(Integer.parseInt(tokens[0]), Integer.parseInt(tokens[1])));
      }
      reader.close();
      return rands;
  }
  
  
  
  public static void writeResultsScatterPlot(String dsName, String resultPath, String dist, ArrayList<String> exactDistNames) throws IOException
  {
	  File summary = new File(resultPath+"/"+dsName+"_approxResults"+".txt");
	  BufferedWriter swriter = new BufferedWriter(new FileWriter(summary));
		
		swriter.write("%" +dsName);
		swriter.newLine();
		
		ArrayList<Double> tempDistances = readResults(resultPath, dsName, dist);
		for(String exactDistName: exactDistNames)
		{
			swriter.write("\\nextgroupplot[title="+dsName+",xlabel="+exactDistName+" Approximation, xmin=100, ymin=100,xmax=250, ymax=250] \n"
					+ "\\addplot[color=lightgray, forget plot] coordinates{(0,0)(260,260) };");
			ArrayList<Double> distances = readResults(resultPath, dsName, exactDistName);
			System.out.println(dist);
			swriter.write("\\addplot[only marks, mark=x, color=blue] \n coordinates{ \n");
			swriter.write("%("+exactDistName+","+dist +") \n");

			
			
			for (int i=0; i< tempDistances.size(); i++) {
					swriter.write("("+String.format(Locale.ENGLISH, "%1.2f",distances.get(i))+","+String.format(Locale.ENGLISH, "%1.2f",tempDistances.get(i))+")");
					swriter.newLine();
			}
			swriter.newLine();
			swriter.write("}; \\label{"+dist +"}");
			swriter.newLine();
			swriter.flush();
		}
		swriter.newLine();
		swriter.close();
  }
  
  public static void evaluateRuntime(ArrayList<GraphDistance> dists, String path, String dsname, ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds, ArrayList<Pair<Integer, Integer>> pairs) throws InterruptedException, IOException
  {
	    BufferedWriter rwriter = new BufferedWriter(new FileWriter(path + "/runtime_results.txt", true));
	    long fulltime = 0;
	    rwriter.write(dsname);
	    rwriter.newLine();
	    for (GraphDistance distance : dists)
		{
			double averagetime = 0;
			for (int i = 0; i < 5; i++) // runtime average over 5 runs
			{
				fulltime = 0;
				for (Pair<Integer, Integer> pair : pairs) {
					long timeStart = System.nanoTime();
					double dist = distance.computeGraphDistance(ds.get(pair.getFirst()), ds.get(pair.getSecond()));
					long timeEnd = System.nanoTime();
					long time = timeEnd - timeStart;
					fulltime += time;
				}

				// convert to millis
				fulltime = fulltime / 1000000;
				averagetime += (double) fulltime / (double) pairs.size();
				System.gc();
				Thread.sleep(2);
				System.gc();
			}
			averagetime = averagetime / 5;
			rwriter.write(distance.getID() + " &");
			rwriter.write(String.format(Locale.ENGLISH, "%1.2f", averagetime) + "\t");
			rwriter.write("\\\\");
			rwriter.newLine();
			rwriter.flush();

		}

		rwriter.close();
  }
  
  
  public static void writeResultsApproximationError(String dsName, String resultPath, String exactDistName, ArrayList<String> distis)
          throws IOException, InterruptedException
  {
    BufferedWriter ewriter = new BufferedWriter(
        new FileWriter(resultPath + "/approximation_results.txt", true));

      ewriter.write(dsName);
      ewriter.newLine();

      ArrayList<Double> cdists = readResults(resultPath, dsName, exactDistName);
     
      // read distances
      for (String distance : distis)
      {
        ArrayList<Double> adists = readResults(resultPath, dsName, distance);
        double score = 0;
        for (int v = 0; v < cdists.size(); v++)
        {
          if (cdists.get(v) == 0 && adists.get(v) != 0)
          {
            System.out.println("Bad approximation!");
          }
          if (cdists.get(v) > 0)
          {
            score += Math.abs(adists.get(v) - cdists.get(v)) / cdists.get(v);
          }
        }
        ewriter.write(distance + " &");
        ewriter
        .write(String.format(Locale.ENGLISH, "%1.2f", score / cdists.size())
            + "\t");
        ewriter.write("\\\\");
        ewriter.newLine();
        ewriter.flush();

    }
    ewriter.close();
  }

 
}