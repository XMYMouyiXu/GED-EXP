package ged.app;

import java.io.IOException;
import java.util.ArrayList;

import org.jgrapht.graph.AbstractGraph;

import ged.Util;
import ged.alg.TestIsomorphic;
import ged.structures.graph.LabeledEdge;
import ged.structures.graph.LabeledVertex;
import ged.structures.graph.PGraph;

public class IsomorphismExperiment
{

  public static void main(String[] args) throws IOException
  {
    String dspath = "data/TUDatasets";
    String[] datasets = {"MUTAG", "PTC_FM"};

    for (String dataset : datasets)
    {
    	System.out.println(dataset);
      ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds = Util
          .readTUDataset(dspath, dataset);
      ds = Util.delabelDataset(ds);
      ArrayList<AbstractGraph<LabeledVertex, LabeledEdge>> dsabs = new ArrayList<AbstractGraph<LabeledVertex, LabeledEdge>>();
      
      for (PGraph<LabeledVertex, LabeledEdge> g : ds) 
      {  
    	  dsabs.add(g);
      }
      
      for (PGraph<LabeledVertex, LabeledEdge> graph : ds)
      {
		for(PGraph<LabeledVertex, LabeledEdge> graph2 :ds)
		{
			boolean wlc = TestIsomorphic.isomorphicWLCompressed(graph, graph2);
			boolean wl = TestIsomorphic.isomorphicWL(graph, graph2);
			boolean v1 = TestIsomorphic.isomorphicIterativeV1(graph, graph2);
			if(wlc != wl)
			{
				System.out.println("wlc and wl not equal "+ graph.getProperty("index") + " " + graph2.getProperty("index")+ " wl: "+wl + " wlc:"+wlc);
			}
			if(v1!=wl)
			{
				System.out.println("v1 and wl not equal "+ graph.getProperty("index") + " " + graph2.getProperty("index")+ " wl: "+wl + " v1:"+v1);
				if(v1)
				{
					System.out.println("Found counterexample on graph level.");
				}
				if(wl)
				{
					System.out.println("Another example where Ntrees are better!");
				}
			}
		}
    	  
      }

    }
    System.out.println("finished!");
  }
}
