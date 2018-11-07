package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Builder for recursive graph traversal
 * 
 * @author Christian Skärby
 */
public class TraverseBuilder<T> {
    
    private final Graph<T> baseGraph;
    
    private Predicate<T> enterCondition = vertex -> true;
    private Predicate<T> traverseCondition;
    private Consumer<T> enterListener = vertex -> {/* ignore */};
    private Consumer<T> leaveListener = vertex -> {/* ignore */};
    private Predicate<T> visitCondition = vertex -> true;
    private Consumer<T> visitListener = vertex -> {/* ignore */};

    /**
     * Constructor
     * @param baseGraph base graph
     */
    public TraverseBuilder(Graph<T> baseGraph) {
        this.baseGraph = baseGraph;
    }

    /**
     * Builder for the standard way to traverse ComputationGraphs in the backward direction
     * @param builder Has the configuration to traverse
     * @return a {@link TraverseBuilder}
     */
    public static TraverseBuilder<String> backwards(ComputationGraphConfiguration.GraphBuilder builder) {
        return new TraverseBuilder<>(new BackwardOf(builder))
                .traverseCondition(GraphBuilderUtil.changeSizePropagates(builder))
                .enterCondition(GraphBuilderUtil.changeSizePropagatesBackwards(builder));
    }

    /**
     * Builder for the standard way to traverse ComputationGraphs in the forward direction
     * @param builder Has the configuration to traverse
     * @return a {@link TraverseBuilder}
     */
    public static TraverseBuilder<String> forwards(ComputationGraphConfiguration.GraphBuilder builder) {
        return new TraverseBuilder<>(new ForwardOf(builder))
                .traverseCondition(GraphBuilderUtil.changeSizePropagates(builder));
    }

    /**
     * Sets a condition for traversal to be entered in the first place. Vertices not matching the condition will
     * return an empty stream.
     *
     * @param enterCondition the condition
     * @return the builder for fluent API
     */
    public TraverseBuilder<T> enterCondition(Predicate<T> enterCondition) {
        this.enterCondition = enterCondition;
        return this;
    }

    /**
     * Set the condition for traversing to the next vertex. Default is if the current vertex
     * is of a type where nOut must be equal to nIn.
     *
     * @param traverseCondition the condition
     * @return the builder for fluent API
     */
    public TraverseBuilder<T> traverseCondition(Predicate<T> traverseCondition) {
        this.traverseCondition = traverseCondition;
        return this;
    }

    /**
     * Set listener to listen for when the scope of a new vertex is entered and its children will be queried. Any
     * subsequent vertices given to visitListener are descendants of this vertex.
     *
     * @param enterListener the listener
     * @return the builder for fluent API
     */
    public TraverseBuilder<T> enterListener(Consumer<T> enterListener) {
        this.enterListener = enterListener;
        return this;
    }

    /**
     * Set listener for when the scope of the a vertex is left. Any subsequent vertices given to visitListener are not
     * descendants of this vertex.
     *
     * @param leaveListener the listener
     * @return the builder for fluent API
     */
    public TraverseBuilder<T> leaveListener(Consumer<T> leaveListener) {
        this.leaveListener = leaveListener;
        return this;
    }

    /**
     * Set a listener for when a vertex is visited. Vertices given to this listener are descendants to any vertex given
     * to enterListener but not yet given to leaveListener.
     *
     * @param visitListener the listener
     * @return the builder for fluent API
     */
    public TraverseBuilder<T> visitListener(Consumer<T> visitListener) {
        this.visitListener = visitListener;
        return this;
    }

    /**
     * Sets a condition for vertices to be visited. Only those vertices matching the condition will be visited.
     *
     * @param visitCondition the condition
     * @return the builder for fluent API
     */
    public TraverseBuilder<T> visitCondition(Predicate<T> visitCondition) {
        this.visitCondition = visitCondition;
        return this;
    }

    /**
     * Build a forward traversing Graph
     *
     * @return a forward traversing Graph
     */
    public Graph<T> build() {
        return
                new EnterIf<>(enterCondition,
                        new Traverse<>(
                                traverseCondition,
                                enterListener,
                                leaveListener,
                                new Peek<>(visitListener,
                                        new Filter<>(visitCondition,
                                                new SingleVisit<>(baseGraph)))));
    }
}
