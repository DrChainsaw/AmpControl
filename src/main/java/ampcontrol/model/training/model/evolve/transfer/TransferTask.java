package ampcontrol.model.training.model.evolve.transfer;

import java.util.Optional;

/**
 * An instruction for how to prune weights of a layer, including inputs to subsequent layers
 */
public interface TransferTask {

    /**
     * Interface for Builders of {@link TransferTask}s which may have dependent tasks
     */
    interface ListBuilder {
        /**
         * Add a {@link ListBuilder} for a dependent task. It can be expected that the build method of the added
         * builder is called when the build method of this builder is called.
         * @param builder {@link ListBuilder} for the dependent task
         * @return The {@link ListBuilder} which the builder was added to (i.e. NOT the added builder)
         */
        ListBuilder addDependentTask(ListBuilder builder);

        /**
         * Builds a {@link TransferTask}, including all dependent tasks.
         * @return The constructed {@link TransferTask}.
         */
        TransferTask build();
    }

    /**
     * Utility class to avoid duplicate handling dependent task builder in all ListBuilders
     */
    class DependentTaskBuilder {

        private Optional<ListBuilder> dependentTaskBuilder = Optional.empty();

        public ListBuilder addDependentTask(ListBuilder builder) {
            if (!this.dependentTaskBuilder.isPresent()) {
                this.dependentTaskBuilder = Optional.of(builder);
            } else {
                this.dependentTaskBuilder.get().addDependentTask(builder);
            }
            return dependentTaskBuilder.get();
        }
    }

    /**
     * Adds element indexes in a given dimension which shall be kept from a source.
     *
     * @param dim     dimension from which elements are wanted
     * @param indexes wanted element indexes in given dimension
     */
    void addWantedElementsFromSource(int dim, int[] indexes);

    /**
     * Adds element indexes in a given dimension which shall be assigned in a target.
     *
     * @param dim     dimension from which elements are wanted
     * @param indexes wanted element indexes in given dimension
     */
    void addWantedElementsFromTarget(int dim, int[] indexes);

    /**
     * Adds element indexes in a given dimension which shall be assigned in a target.
     * TODO: Update method signature to use int[] when updating dl4j to include fix for #6327
     *
     * @param dim     dimension from which elements are wanted
     * @param nrofElements wanted number of element indexes in given dimension
     */
    void addWantedNrofElementsFromTarget(int dim, int nrofElements);

    /**
     * Executes the transfer task
     */
    void execute();

}
