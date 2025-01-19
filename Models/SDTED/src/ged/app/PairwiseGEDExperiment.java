package ged.app;

import java.io.IOException;
import java.util.ArrayList;

import org.jgrapht.graph.AbstractGraph;

import ged.Util;
import ged.WritePairwiseGED;
import ged.alg.distance.GraphDistance;
import ged.alg.distance.SDTEDbasedDistanceV1;
import ged.alg.distance.SDTEDbasedDistanceV2;
import ged.alg.distance.SDTEDbasedDistanceWL;
import ged.alg.distance.comparison.BipartiteGraphMatching;
import ged.structures.graph.LabeledEdge;
import ged.structures.graph.LabeledVertex;
import ged.structures.graph.PGraph;

public class PairwiseGEDExperiment
{

  public static void main(String[] args) throws IOException
  {
    String path = "results";
    String dspath = "data/TUDatasets";
    String[] datasets = {"MUTAG"};
    ArrayList<GraphDistance> dists = new ArrayList<GraphDistance>();
    dists.add(new BipartiteGraphMatching());
    dists.add(new SDTEDbasedDistanceV1(3));
    dists.add(new SDTEDbasedDistanceV2(3));
    dists.add(new SDTEDbasedDistanceWL(3));
//    dists.add(new SDTEDbasedDistanceV1(3,0.5));
//	dists.add(new SDTEDbasedDistanceV2(3,0.5));
//	dists.add(new SDTEDbasedDistanceWL(3,0.5));

    for (String dataset : datasets)
    {
    	System.out.println(dataset);
      ArrayList<PGraph<LabeledVertex, LabeledEdge>> ds = Util
          .readTUDataset(dspath, dataset);
      ArrayList<AbstractGraph<LabeledVertex, LabeledEdge>> dsabs = new ArrayList<AbstractGraph<LabeledVertex, LabeledEdge>>();
      for (PGraph<LabeledVertex, LabeledEdge> g : ds)
      {
        dsabs.add(g);
      }
      for (GraphDistance distance : dists)
      {
        WritePairwiseGED.writePairwiseGED(dsabs, distance, path, dataset);
      }
    }
  }
}
