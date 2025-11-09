package edu.uno.ai.planning.ex;

import edu.uno.ai.SearchBudget;
import edu.uno.ai.planning.ss.StateSpacePlanner;
import edu.uno.ai.planning.ss.StateSpaceProblem;
import edu.uno.ai.planning.ss.StateSpaceSearch;

/**
 * A planner that uses simple breadth first search through the space of states.
 * 
 * @author Your Name
 */
public class ExamplePlanner extends StateSpacePlanner {

	/**
	 * Constructs a new breadth first search planner. You should change the
	 * string below from "Example" to your ID. You should also change the name
	 * of this class. In Eclipse, you can do that easily by right-clicking on
	 * this file (ExamplePlanner.java) in the Package Explorer and choosing
	 * Refactor > Rename.
	 */
	public ExamplePlanner() {
		super("Example");
	}

	@Override
	protected StateSpaceSearch makeStateSpaceSearch(StateSpaceProblem problem, SearchBudget budget) {
		return new BreadthFirstSearch(problem, budget);
	}
}
