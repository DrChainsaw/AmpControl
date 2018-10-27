package ampcontrol.model.training.model.evolve.transfer;

import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;

/**
 * {@link TransferTask} with callback for when wanted elements are added. Main use case is to break circular dependencies
 * in {@link MergeTransferBuffer}.
 *
 * @author Christian Skärby
 */
public class CallbackTransferTask implements TransferTask {

    private final TransferTask dependentTask;
    private final BiConsumer<Integer, int[]> sourceCallback;
    private final BiConsumer<Integer, int[]> targetCallback;

    /**
     * Create a builder for this class
     *
     * @return a new {@link Builder} instance.
     */
    public static Builder builder() {
        return new Builder();
    }

    public CallbackTransferTask(
            TransferTask dependentTask,
            BiConsumer<Integer, int[]> sourceCallback,
            BiConsumer<Integer, int[]> targetCallback) {
        this.dependentTask = dependentTask;
        this.sourceCallback = sourceCallback;
        this.targetCallback = targetCallback;
    }

    @Override
    public void addWantedElementsFromSource(int dim, int[] indexes) {
        sourceCallback.accept(dim, indexes);
        dependentTask.addWantedElementsFromSource(dim, indexes);
    }

    @Override
    public void addWantedElementsFromTarget(int dim, int[] indexes) {
        targetCallback.accept(dim, indexes);
        dependentTask.addWantedElementsFromTarget(dim, indexes);

    }

    @Override
    public void addWantedNrofElementsFromTarget(int dim, int nrofElements) {
        targetCallback.accept(dim,  IntStream.range(0, nrofElements).toArray());
        dependentTask.addWantedNrofElementsFromTarget(dim, nrofElements);
    }

    @Override
    public void execute() {
        throw new UnsupportedOperationException("Shall never execute!");
    }

    public static class Builder implements ListBuilder {

        private Optional<ListBuilder> dependentTaskBuilder = Optional.empty();
        private CallbackTransferTask instance;
        private BiConsumer<Integer, int[]> sourceCallback;
        private BiConsumer<Integer, int[]> targetCallback;

        public Builder setSourceCallback(BiConsumer<Integer, int[]> sourceCallback) {
            this.sourceCallback = sourceCallback;
            return this;
        }

        public Builder setTargetCallback(BiConsumer<Integer, int[]> targetCallback) {
            this.targetCallback = targetCallback;
            return this;
        }

        @Override
        public ListBuilder addDependentTask(ListBuilder dependentTaskBuilder) {
            if (!this.dependentTaskBuilder.isPresent()) {
                this.dependentTaskBuilder = Optional.of(dependentTaskBuilder);
            } else {
                this.dependentTaskBuilder.get().addDependentTask(dependentTaskBuilder);
            }
            return this;
        }

        @Override
        public CallbackTransferTask build() {
            return Optional.ofNullable(instance).orElseGet(() -> {
                instance = new CallbackTransferTask(
                        dependentTaskBuilder.map(ListBuilder::build).orElse(new NoTransferTask()),
                        sourceCallback,
                        targetCallback);
                return instance;
            });
        }
    }
}
