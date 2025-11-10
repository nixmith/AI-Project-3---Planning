package edu.uno.ai.planning.ex;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.Writer;

import edu.uno.ai.planning.Domain;
import edu.uno.ai.planning.Main;
import edu.uno.ai.planning.Planner;
import edu.uno.ai.planning.Problem;
import edu.uno.ai.planning.Settings;
import edu.uno.ai.planning.bb.BlackBox;
import edu.uno.ai.planning.bfs.BreadthFirstSearchPlanner;
import edu.uno.ai.planning.gp.GraphPlan;
import edu.uno.ai.planning.hsp.HeuristicSearchPlanner;
import edu.uno.ai.planning.io.PlanningParser;
import edu.uno.ai.planning.pop.PartialOrderPlanner;

public class Test {
	
	private static final Planner<?>[] PLANNERS = new Planner[] {
		new NicksPlanner(),
		new BreadthFirstSearchPlanner(),
		new PartialOrderPlanner(),
		new GraphPlan(),
		new BlackBox(),
		new HeuristicSearchPlanner()
	};

	private static final String[] DOMAINS = new String[] {
		"benchmarks/cake.pddl",
		"benchmarks/blocks.pddl",
		"benchmarks/cargo.pddl",
		"benchmarks/wumpus.pddl",
	};

	private static final String[] PROBLEMS = new String[] {
		"benchmarks/do_nothing.pddl",
		"benchmarks/eat_cake.pddl",
		"benchmarks/have_eat_cake.pddl",
		"benchmarks/easy_stack.pddl",
		"benchmarks/easy_unstack.pddl",
		"benchmarks/sussman.pddl",
		"benchmarks/reverse_2.pddl",
		"benchmarks/reverse_4.pddl",
		"benchmarks/reverse_6.pddl",
		"benchmarks/reverse_8.pddl",
		"benchmarks/reverse_10.pddl",
		"benchmarks/reverse_12.pddl",
		"benchmarks/reverse_14.pddl",
		"benchmarks/deliver_1.pddl",
		"benchmarks/deliver_2.pddl",
		"benchmarks/deliver_3.pddl",
		"benchmarks/deliver_4.pddl",
		"benchmarks/deliver_5.pddl",
		"benchmarks/deliver_return_1.pddl",
		"benchmarks/deliver_return_2.pddl",
		"benchmarks/deliver_return_3.pddl",
		"benchmarks/deliver_return_4.pddl",
		"benchmarks/deliver_return_5.pddl",
		"benchmarks/easy_wumpus.pddl",
		"benchmarks/medium_wumpus.pddl",
		"benchmarks/hard_wumpus.pddl",
	};
	
	public static void main(String[] args) throws Exception {
		PlanningParser parser = new PlanningParser();
		for(String url : DOMAINS) {
			System.out.println("Reading domain \"" + url + "\"...");
			Domain domain = parser.parse(new File(url), Domain.class);
			parser.setDefined(domain.name, domain);
		}
		Problem[] problems = new Problem[PROBLEMS.length];
		for(int i=0; i<problems.length; i++) {
			System.out.println("Reading problem \"" + PROBLEMS[i] + "\"...");
			problems[i] = parser.parse(new File(PROBLEMS[i]), Problem.class);
		}
		try(Writer output = new BufferedWriter(new FileWriter("results.html"))) {
			Main.benchmark(PLANNERS, problems, Settings.OPERATIONS_LIMIT, Settings.TIME_LIMIT, output);
		}
	}
}
