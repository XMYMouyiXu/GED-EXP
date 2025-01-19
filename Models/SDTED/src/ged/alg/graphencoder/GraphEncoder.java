package ged.alg.graphencoder;

import org.jgrapht.graph.AbstractGraph;

public abstract class GraphEncoder<V, E>
{
  protected final AbstractGraph<V, E> graph;
  protected GraphEncoder(AbstractGraph<V, E> graph)
  {
    this.graph = graph;
  }

  public abstract boolean refine();

  public void fullyRefine()
  {
    while (refine());
  }
  /**
 * @param refinementsteps maximum number of refinement steps (possibly less if graph is smaller), <0: fully refine
 */
  public void refinementSteps(int refinementsteps)
  {
	  if(refinementsteps <0)
	    {
	    	fullyRefine();
	    }
	    else
	    {
	    	for(int i =0; i<refinementsteps;i++)
	    	{
	    		boolean refined = refine();
	    		if(!refined)
	    		{
	    			break;
	    		}
	    	}	
	    }
  }

  
  public abstract String computeLexicographicEncoding();
}
