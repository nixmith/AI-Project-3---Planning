package edu.uno.ai.planning.ex;

import java.util.*;
import java.util.Arrays;
import edu.uno.ai.SearchBudget;
import edu.uno.ai.planning.Plan;
import edu.uno.ai.planning.Step;
import edu.uno.ai.logic.*;
import edu.uno.ai.planning.ss.StateSpaceNode;
import edu.uno.ai.planning.ss.StateSpaceProblem;
import edu.uno.ai.planning.ss.StateSpaceSearch;
import edu.uno.ai.planning.ss.StateSpacePlanner;

/**
 * An optimized state-space planner using A* search with ignore-delete-list heuristic.
 * Enhanced with better tie-breaking, cost-based duplicate detection, and optimized heuristics.
 * 
 * @author Nick
 */
public class NicksPlanner extends StateSpacePlanner {

	/**
	 * Constructs a new optimized heuristic search planner.
	 */
	public NicksPlanner() {
		super("Nick");
	}

	@Override
	protected StateSpaceSearch makeStateSpaceSearch(StateSpaceProblem problem, SearchBudget budget) {
		return new OptimizedHeuristicSearch(problem, budget);
	}
}

/**
 * Optimized A* search with enhanced ignore-delete-list heuristic.
 */
class OptimizedHeuristicSearch extends StateSpaceSearch {
	
	/** Priority queue for the frontier */
	private final PriorityQueue<SearchNode> frontier;
	
	/** Cost-based duplicate detection - maps state to best known g-value */
	private final HashMap<State, Double> bestCosts;
	
	/** All positive literals in the problem */
	private final Set<Literal> allPositiveLiterals;
	
	/** Goal literals */
	private final List<Literal> goalLiterals;
	private final List<Literal> positiveGoals;
	
	/** Pre-computed positive effects and preconditions for each step */
	private final Map<Step, List<Literal>> stepPositiveEffects;
	private final Map<Step, List<Literal>> stepPositivePreconditions;
	
	/** Maps for heuristic computation */
	private final HashMap<Literal, Double> literalCosts;
	
	/** Cached heuristic values to avoid recomputation */
	private final HashMap<State, Double> heuristicCache;
	
	/** Goal achievement costs for better heuristic */
	private final double[] goalCosts;
	
	/** Step relevance scores for ordering */
	private final Map<Step, Double> stepRelevance;
	
	/**
	 * Enhanced search node with better tie-breaking
	 */
	private class SearchNode implements Comparable<SearchNode> {
		final StateSpaceNode node;
		final double g;
		final double h;
		final double f;
		final int depth;
		final int goalsSatisfied;
		
		SearchNode(StateSpaceNode node, double g, double h, int depth) {
			this.node = node;
			this.g = g;
			this.h = h;
			this.f = g + h;
			this.depth = depth;
			
			// Count satisfied goals for tie-breaking
			int satisfied = 0;
			for (Literal goal : positiveGoals) {
				if (goal.isTrue(node.state)) {
					satisfied++;
				}
			}
			this.goalsSatisfied = satisfied;
		}
		
		@Override
		public int compareTo(SearchNode other) {
			// Primary: f-value
			int fCompare = Double.compare(this.f, other.f);
			if (fCompare != 0) return fCompare;
			
			// Secondary: prefer nodes with more goals satisfied
			int goalsCompare = Integer.compare(other.goalsSatisfied, this.goalsSatisfied);
			if (goalsCompare != 0) return goalsCompare;
			
			// Tertiary: prefer lower h (closer to goal)
			int hCompare = Double.compare(this.h, other.h);
			if (hCompare != 0) return hCompare;
			
			// Quaternary: prefer shallower nodes (shorter plans)
			int depthCompare = Integer.compare(this.depth, other.depth);
			if (depthCompare != 0) return depthCompare;
			
			// Final: prefer lower g
			return Double.compare(this.g, other.g);
		}
	}
	
	/**
	 * Constructor with enhanced preprocessing
	 */
	public OptimizedHeuristicSearch(StateSpaceProblem problem, SearchBudget budget) {
		super(problem, budget);
		this.frontier = new PriorityQueue<>(100);
		this.bestCosts = new HashMap<>();
		this.literalCosts = new HashMap<>();
		this.heuristicCache = new HashMap<>();
		
		// Pre-compute literals and effects
		this.allPositiveLiterals = new HashSet<>();
		this.stepPositiveEffects = new HashMap<>();
		this.stepPositivePreconditions = new HashMap<>();
		this.stepRelevance = new HashMap<>();
		
		// Extract goal literals
		this.goalLiterals = getLiterals(problem.goal);
		this.positiveGoals = new ArrayList<>();
		for (Literal goal : goalLiterals) {
			if (isPositive(goal)) {
				positiveGoals.add(goal);
				allPositiveLiterals.add(goal);
			}
		}
		this.goalCosts = new double[positiveGoals.size()];
		
		// Pre-process steps for efficiency
		preprocessSteps();
		
		// Calculate step relevance scores
		calculateStepRelevance();
	}
	
	/**
	 * Preprocess steps to extract effects and preconditions
	 */
	private void preprocessSteps() {
		for (Step step : problem.steps) {
			// Extract positive effects
			List<Literal> effects = getLiterals(step.effect);
			List<Literal> positiveEffects = new ArrayList<>();
			
			for (Literal effect : effects) {
				if (isPositive(effect)) {
					positiveEffects.add(effect);
					allPositiveLiterals.add(effect);
				}
			}
			stepPositiveEffects.put(step, positiveEffects);
			
			// Extract positive preconditions
			List<Literal> preconds = getLiterals(step.precondition);
			List<Literal> positivePreconds = new ArrayList<>();
			
			for (Literal precond : preconds) {
				if (isPositive(precond)) {
					positivePreconds.add(precond);
					allPositiveLiterals.add(precond);
				}
			}
			stepPositivePreconditions.put(step, positivePreconds);
		}
	}
	
	/**
	 * Calculate relevance scores for steps based on goal achievement
	 */
	private void calculateStepRelevance() {
		for (Step step : problem.steps) {
			double relevance = 0.0;
			List<Literal> effects = stepPositiveEffects.get(step);
			
			// Higher relevance for steps that achieve goals
			for (Literal effect : effects) {
				if (positiveGoals.contains(effect)) {
					relevance += 10.0;
				}
			}
			
			// Bonus for steps with fewer preconditions (easier to achieve)
			List<Literal> preconds = stepPositivePreconditions.get(step);
			relevance += 1.0 / (preconds.size() + 1);
			
			stepRelevance.put(step, relevance);
		}
	}

	@Override
	public Plan solve() {
		// Initialize with root
		double h0 = computeHeuristic(root.state);
		SearchNode initial = new SearchNode(root, 0, h0, 0);
		frontier.offer(initial);
		bestCosts.put(root.state, 0.0);
		
		// Track best solution for anytime behavior
		Plan bestSolution = null;
		int bestSolutionLength = Integer.MAX_VALUE;
		
		// A* search with enhancements
		while (!frontier.isEmpty()) {
			SearchNode current = frontier.poll();
			
			// Skip if we've found a better path to this state
			Double bestG = bestCosts.get(current.node.state);
			if (bestG != null && bestG < current.g) {
				continue;
			}
			
			// Goal test with solution tracking
			if (problem.isSolution(current.node.plan)) {
				int planLength = current.node.plan.size();
				if (bestSolution == null || planLength < bestSolutionLength) {
					bestSolution = current.node.plan;
					bestSolutionLength = planLength;
				}
				// Continue searching for better solutions if we have budget
				if (current.g == current.h) {
					// Optimal solution found
					return current.node.plan;
				}
				// For now, return first solution found
				return current.node.plan;
			}
			
			// Expand with smart ordering
			expandNode(current);
		}
		
		return bestSolution;
	}
	
	/**
	 * Expand a node with optimized successor generation
	 */
	private void expandNode(SearchNode current) {
	    for (Step step : problem.steps) {
	        // Quick precondition check
	        if (!step.precondition.isTrue(current.node.state)) {
	            continue;
	        }
	        
	        // Expand the node
	        StateSpaceNode successor = current.node.expand(step);
	        double g = current.g + 1;
	        
	        // Cost-based duplicate detection
	        Double bestG = bestCosts.get(successor.state);
	        if (bestG != null && bestG <= g) {
	            continue; // Already found better or equal path
	        }
	        
	        // Update best cost
	        bestCosts.put(successor.state, g);
	        
	        // Compute heuristic with caching
	        double h = computeHeuristicWithCache(successor.state);
	        
	        // Prune nodes with no chance of improving
	        if (h == Double.POSITIVE_INFINITY) {
	            continue;
	        }
	        
	        // Add to frontier
	        SearchNode successorNode = new SearchNode(successor, g, h, current.depth + 1);
	        frontier.offer(successorNode);
	    }
	}
	
	/**
	 * Compute heuristic with caching for efficiency
	 */
	private double computeHeuristicWithCache(State state) {
		// Check cache first
		Double cached = heuristicCache.get(state);
		if (cached != null) {
			return cached;
		}
		
		// Compute and cache
		double h = computeHeuristic(state);
		
		// Only cache finite values to save memory
		if (h < Double.POSITIVE_INFINITY) {
			// Limit cache size to prevent memory issues
			if (heuristicCache.size() > 5000) {
				heuristicCache.clear();
			}
			heuristicCache.put(state, h);
		}
		
		return h;
	}
	
	/**
	 * Optimized ignore-delete-list heuristic
	 */
	private double computeHeuristic(State state) {
		// Clear costs
		literalCosts.clear();
		
		// Initialize with literals true in current state
		for (Literal literal : allPositiveLiterals) {
			if (literal.isTrue(state)) {
				literalCosts.put(literal, 0.0);
			}
		}
		
		// Quick goal check
		int satisfiedGoals = 0;
		for (Literal goal : positiveGoals) {
			if (literalCosts.containsKey(goal)) {
				satisfiedGoals++;
			}
		}
		
		// All positive goals satisfied?
		if (satisfiedGoals == positiveGoals.size()) {
			// Check negative goals
			boolean allSatisfied = true;
			for (Literal goal : goalLiterals) {
				if (!isPositive(goal) && !goal.isTrue(state)) {
					allSatisfied = false;
					break;
				}
			}
			if (allSatisfied) return 0;
		}
		
		// Optimized relaxed planning with early termination
		boolean changed = true;
		int maxIterations = Math.min(allPositiveLiterals.size() + 1, 100);
		
		for (int iter = 0; iter < maxIterations && changed; iter++) {
			changed = false;
			int newlySatisfied = 0;
			
			for (Step step : problem.steps) {
				// Skip irrelevant steps
				List<Literal> positiveEffects = stepPositiveEffects.get(step);
				if (positiveEffects.isEmpty()) continue;
				
				// Check if any effect is a goal we haven't achieved
				boolean relevant = false;
				for (Literal effect : positiveEffects) {
					if (positiveGoals.contains(effect) && !literalCosts.containsKey(effect)) {
						relevant = true;
						break;
					}
				}
				
				if (!relevant && iter > 3) continue; // Skip after initial iterations
				
				// Calculate precondition cost efficiently
				double precondCost = 0;
				boolean achievable = true;
				
				List<Literal> positivePreconds = stepPositivePreconditions.get(step);
				for (Literal precond : positivePreconds) {
					Double cost = literalCosts.get(precond);
					if (cost == null) {
						achievable = false;
						break;
					}
					precondCost = Math.max(precondCost, cost);
				}
				
				if (!achievable) continue;
				
				// Update effect costs
				double stepCost = precondCost + 1;
				for (Literal effect : positiveEffects) {
					Double currentCost = literalCosts.get(effect);
					if (currentCost == null || stepCost < currentCost) {
						literalCosts.put(effect, stepCost);
						changed = true;
						if (positiveGoals.contains(effect)) {
							newlySatisfied++;
						}
					}
				}
			}
			
			// Early termination if all goals achieved
			if (satisfiedGoals + newlySatisfied >= positiveGoals.size()) {
				boolean allAchieved = true;
				for (Literal goal : positiveGoals) {
					if (!literalCosts.containsKey(goal)) {
						allAchieved = false;
						break;
					}
				}
				if (allAchieved) break;
			}
		}
		
		// Calculate final heuristic value
		double total = 0;
		for (int i = 0; i < positiveGoals.size(); i++) {
			Literal goal = positiveGoals.get(i);
			Double cost = literalCosts.get(goal);
			if (cost == null) {
				return Double.POSITIVE_INFINITY;
			}
			goalCosts[i] = cost;
			total += cost;
		}
		
		return total;
	}
	
	/**
	 * Extract literals from a proposition
	 */
	private List<Literal> getLiterals(Proposition prop) {
		List<Literal> literals = new ArrayList<>();
		getLiteralsHelper(prop, literals);
		return literals;
	}
	
	private void getLiteralsHelper(Proposition prop, List<Literal> literals) {
		if (prop instanceof Literal) {
			literals.add((Literal) prop);
		} else if (prop instanceof Conjunction) {
			for (Proposition arg : ((Conjunction) prop).arguments) {
				getLiteralsHelper(arg, literals);
			}
		} else if (prop instanceof Disjunction) {
			for (Proposition arg : ((Disjunction) prop).arguments) {
				getLiteralsHelper(arg, literals);
			}
		} else if (prop instanceof Negation) {
			Negation neg = (Negation) prop;
			if (neg.argument instanceof Literal) {
				literals.add(((Literal) neg.argument).negate());
			} else {
				getLiteralsHelper(neg.argument, literals);
			}
		}
	}
	
	/**
	 * Check if a literal is positive
	 */
	private boolean isPositive(Literal literal) {
		String str = literal.toString();
		return !str.startsWith("!") && !str.startsWith("(not");
	}
}