package ampcontrol.model.training.model.evolve.transfer;

/**
 * No-op typically used to terminate a chain of dependent {@link TransferTask}s
 *
 * @author Christian Skärby
 */
public class NoTransferTask implements TransferTask {

    /**
     * Create a builder for this class
     *
     * @return a new {@link MergeTransferBuffer.Builder} instance.
     */
    public static Builder builder() {
        return new Builder();
    }

    @Override
    public void addWantedElementsFromSource(int dim, int[] indexes) {
        // Ignore
    }

    @Override
    public void addWantedElementsFromTarget(int dim, int[] indexes) {
        // Ignore
    }

    @Override
    public void addWantedNrofElementsFromTarget(int dim, int nrofElements) {
        // Ignore
    }

    @Override
    public void execute() {
        // Ignore
    }

    /**
     * Builder just to fit into the API.
     */
    private static class Builder implements ListBuilder {

        /**
         * Returns the given ListBuilder instead of itself, essentially replacing itself with the given builder.
         * @param builder {@link ListBuilder} for the dependent task
         * @return the given ListBuilder.
         */
        @Override
        public ListBuilder addDependentTask(ListBuilder builder) {
            return builder;
        }

        @Override
        public TransferTask build() {
            return new NoTransferTask();
        }
    }
}
