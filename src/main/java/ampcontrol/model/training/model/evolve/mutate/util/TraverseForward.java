package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Builder for the standard way to traverse ComputationGraphs in the forward direction
 *
 * @author Christian Skärby
 */
public class TraverseForward {

    private final Graph<String> baseGraph;

    private Predicate<String> traverseCondition;
    private Consumer<String> enterListener = vertex -> {/* ignore */};
    private Consumer<String> leaveListener = vertex -> {/* ignore */};
    private Predicate<String> visitCondition = vertex -> true;
    private Consumer<String> visitListener = vertex -> {/* ignore */};

    public TraverseForward(ComputationGraphConfiguration.GraphBuilder builder) {
        baseGraph = new ForwardOf(builder);
        traverseCondition = GraphBuilderUtil.changeSizePropagates(builder);
    }

    /**
     * Set the condition for traversing to the next vertex. Default is if the current vertex
     * is of a type where nOut must be equal to nIn.
     *
     * @param traverseCondition the condition
     * @return the builder for fluent API
     */
    public TraverseForward traverseCondition(Predicate<String> traverseCondition) {
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
    public TraverseForward enterListener(Consumer<String> enterListener) {
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
    public TraverseForward leaveListener(Consumer<String> leaveListener) {
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
    public TraverseForward visitListener(Consumer<String> visitListener) {
        this.visitListener = visitListener;
        return this;
    }

    /**
     * Sets a condition for vertices to be visited. Only those vertices matching the condition will be visited.
     *
     * @param visitCondition the condition
     * @return the builder for fluent API
     */
    public TraverseForward visitCondition(Predicate<String> visitCondition) {
        this.visitCondition = visitCondition;
        return this;
    }

    /**
     * Build a forward traversing Graph
     *
     * @return a forward traversing Graph
     */
    public Graph<String> build() {
        return new Traverse<>(
                traverseCondition,
                enterListener,
                leaveListener,
                new Peek<>(visitListener,
                        new Filter<>(visitCondition,
                                new SingleVisit<>(baseGraph))));
    }
}
