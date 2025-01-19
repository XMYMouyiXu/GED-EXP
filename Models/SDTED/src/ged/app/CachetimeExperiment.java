package ged.app;
import java.io.BufferedWriter;
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
import ged.structures.graph.LabeledEdge;
import ged.structures.graph.LabeledVertex;
import ged.structures.graph.PGraph;

public class CachetimeExperiment {

  public static void main(String[] args)
      throws IOException, InterruptedException
  {
    String path = "results";
    String dspath = "data/TUDatasets";
    String[] datasets = {"proteincom_sampled","MUTAG", "PTC_FM","Letter-med","MSRC_9","NCI1"};
    ArrayList<GraphDistance> dists = new ArrayList<GraphDistance>();
    double weight = 0.5;
    for (int i = 1; i <= 10; i++)
    {
      dists.add(new SDTEDbasedDistanceV1(i, weight));
    }

    for (int i = 1; i <= 10; i++)
    {
        dists.add(new SDTEDbasedDistanceV2(i, weight));
    }

    for (int i = 1; i <= 10; i++)
    {
        dists.add(new SDTEDbasedDistanceWL(i, weight));
    }
    
    
    //// not cached
    
    for (int i = 1; i <= 10; i++)
    {
    	  SDTEDbasedDistanceV1 d = new SDTEDbasedDistanceV1(i, weight);
    	  d.setCachedAssignment(false);
    	  dists.add(d);
    }

    for (int i = 1; i <= 10; i++)
    {
    	  SDTEDbasedDistanceV2 d = new SDTEDbasedDistanceV2(i, weight);
    	  d.setCachedAssignment(false);
    	  dists.add(d);
    }

    for (int i = 1; i <= 5; i++)
    {
    	  SDTEDbasedDistanceWL d = new SDTEDbasedDistanceWL(i, weight);
    	  d.setCachedAssignment(false);
    	  dists.add(d);
    }
    
    
    int n = 20;
    int seed = 42;
    runtime(dists, dspath, path, datasets, n, seed);

  }

  public static void runtime(ArrayList<GraphDistance> dists, String dspath,
      String path, String[] datasets, int n, int seed)
      throws IOException, InterruptedException
  {
    BufferedWriter rwriter = new BufferedWriter(
        new FileWriter(path + "/cachetime_results.txt", true));
    long fulltime = 0;
    for (String dataset : datasets)
    {
      System.out.println(dataset);
      ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds = Util
          .readTUDataset(dspath, dataset);

      rwriter.write(dataset);
      rwriter.newLine();

      // choose random graphs
      ArrayList<Pair<Integer, Integer>> rands = new ArrayList<Pair<Integer, Integer>>();
      Random r = new Random(seed);
      for (int i = 0; i < n; i++)
      {
        rands.add(new Pair<Integer, Integer>(r.nextInt(ds.size()),
            r.nextInt(ds.size())));
      }
      
      int counter = 1;
      for (GraphDistance distance : dists)
      {
        double averagetime = 0;
        for (int i = 0; i < 5; i++) // runtime average over 5 runs
        {
          fulltime = 0;
          for (Pair<Integer, Integer> pair : rands)
          {
            long timeStart = System.nanoTime();
            double dist = distance.computeGraphDistance(ds.get(pair.getFirst()),
                ds.get(pair.getSecond()));
            long timeEnd = System.nanoTime();
            long time = timeEnd - timeStart;
            fulltime += time;
          }

          // convert to millis
          fulltime = fulltime / 1000000;
          averagetime += (double) fulltime / (double) rands.size();
          System.gc();
          Thread.sleep(2);
          System.gc();
        }
        averagetime = averagetime / 5;
        if(counter % 10 == 1)
        {
        	rwriter.write(distance.getID() + " &");
        }
        else {
        	rwriter.write(" &");
        }
        rwriter
            .write(String.format(Locale.ENGLISH, "%1.2f", averagetime) + "\t");
        if(counter % 10 == 0)
        {
        	rwriter.write("\\\\");
        	rwriter.newLine();
        }
        counter++;
        rwriter.flush();
      }
    }
    rwriter.close();
  }
}