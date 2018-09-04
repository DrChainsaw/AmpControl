package ampcontrol.model.training.model.mutate.reshape;

import java.util.Comparator;

/**
 * An instruction for how to prune weights of a layer, including inputs to subsequent layers
 */
public interface ReshapeSubTask {

    /**
     * Adds element indexes in a given dimension which shall be kept from a source.
     *
     * @param dim     dimension from which elements are wanted
     * @param indexes wanted element indexes in given dimension
     */
    void addWantedElementsFromSource(int dim, int[] indexes);

    /**
     * Adds element indexes in a given dimension which shall be kept from a source.
     *
     * @param dim     dimension from which elements are wanted
     * @param nrofElements wanted number of element indexes in given dimension
     */
    void addWantedNrofElementsFromTarget(int dim, int nrofElements);

    /**
     * Decides how to compare elements of the given dimensions
     *
     * @param tensorDimensions dimensions to compare
     * @return a Comparator which can compare elements of the given dimensions
     */
    Comparator<Integer> getComparator(int[] tensorDimensions);

    /**
     * Executes the transfer task
     */
    void execute();

}
