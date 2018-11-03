package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;

import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Builder for the standard way to traverse ComputationGraphs in the backward direction
 *
 * @author Christian Skärby
 */
public class TraverseBackward {

    private final Graph<String> baseGraph;

    private Predicate<String> enterCondition;
    private Predicate<String> traverseCondition;
    private Predicate<String> visitCondition = vertex -> true;
    private Consumer<String> enterListener = vertex -> {};
    private Consumer<String> leaveListener = vertex -> {};
    private Consumer<String> visitListener = vertex -> {};

    public TraverseBackward(ComputationGraphConfiguration.GraphBuilder builder) {
        baseGraph = new BackwardOf(builder);
        traverseCondition = GraphBuilderUtil.changeSizePropagates(builder);
        enterCondition = vertex -> Optional.ofNullable(builder.getVertices().get(vertex))
                .map(graphVertex -> graphVertex instanceof ElementWiseVertex)
                .orElseThrow(() -> new IllegalArgumentException("Unkown vertex name: " + vertex + "!"));
    }

    /**
     * Set the condition for traversing to the next vertex. Default is if the current vertex
     * is of a type where nOut must be equal to nIn.
     * @param traverseCondition the condition
     * @return the builder for fluent API
     */
    public TraverseBackward traverseCondition(Predicate<String> traverseCondition) {
        this.traverseCondition = traverseCondition;
        return this;
    }

    /**
     * Set listener to listen for when the scope of a new vertex is entered and its children will be queried. Any
     * subsequent vertices given to visitListener are ancestors of this node.
     * @param enterListener the listener
     * @return the builder for fluent API
     */
    public TraverseBackward enterListener(Consumer<String> enterListener) {
        this.enterListener = enterListener;
        return this;
    }

    /**
     * Set listener for when the scope of the a vertex is left. Any subsequent vertices given to visitListener are not
     * ancestors of this node.
     * @param leaveListener the listener
     * @return the builder for fluent API
     */
    public TraverseBackward leaveListener(Consumer<String> leaveListener) {
        this.leaveListener = leaveListener;
        return this;
    }

    /**
     * Set a listener for when a vertex is visited. Vertices given to this listener are ancestors to any vertex given
     * to enterListener but not yet given to leaveListener.
     * @param visitListener the listener
     * @return the builder for fluent API
     */
    public TraverseBackward visitListener(Consumer<String> visitListener) {
        this.visitListener = visitListener;
        return this;
    }

    /**
     * Sets a condition for vertices to be visited. Only those vertices matching the condition will be visited.
     * @param visitCondition the condition
     * @return the builder for fluent API
     */
    public TraverseBackward visitCondition(Predicate<String> visitCondition) {
        this.visitCondition = visitCondition;
        return this;
    }

    public Graph<String> build() {
        return  new EnterIf<>(enterCondition,
                new Traverse<>(
                        traverseCondition,
                        enterListener,
                        leaveListener,
                        new Peek<>(visitListener,
                                new SingleVisit<>(
                                        new Filter<>(visitCondition,
                                                baseGraph)))));
    }
}
