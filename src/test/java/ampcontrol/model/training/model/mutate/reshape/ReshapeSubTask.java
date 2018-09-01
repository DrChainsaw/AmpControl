package ampcontrol.model.training.model.mutate.reshape;

import java.util.Comparator;

/**
 * An instruction for how to prune weights of a layer, including inputs to subsequent layers
 */
public interface ReshapeSubTask {
    /**
     * Adds element indexes in a given dimension which shall be kept from a source.
     *
     * @param dim               wanted dimension
     * @param wantedElementInds wanted element indexes in given dimension
     */
    void addWantedElements(int dim, int[] wantedElementInds);

    /**
     * Decides how to compare elements of the given dimensions
     *
     * @param tensorDimensions dimensions to compare
     * @return a Comparator which can compare elements of the given dimensions
     */
    Comparator<Integer> getComparator(int[] tensorDimensions);

    /**
     * Assigns the pruned weights
     */
    void assign();
}
