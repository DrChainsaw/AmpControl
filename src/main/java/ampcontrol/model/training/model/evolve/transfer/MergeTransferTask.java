package ampcontrol.model.training.model.evolve.transfer;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * {@link TransferTask} used when outputs are merged for dependent tasks.
 *
 * @author Christian Skärby
 */
public final class MergeTransferTask implements TransferTask {

    private final static class DimIndexBuffer {
        private final int dimension;
        private final List<Integer> offsets;
        private Map<Integer, int[]> indexes = new LinkedHashMap<>();

        private DimIndexBuffer(int dimension) {
            this.dimension = dimension;
            offsets = new ArrayList<>();
            offsets.add(0);
        }

        private void addOffset(long anOffset) {
            offsets.add((int) anOffset);
        }

        private void addIndexes(int ordinal, int[] indexes) {
            this.indexes.put(ordinal, indexes);
        }

        private Optional<int[]> getAllIndexes() {
            if (indexes.isEmpty()) {
                return Optional.empty();
            }

            return Optional.of(
                    IntStream.range(0, offsets.size()-1).flatMap(ordinal ->
                            IntStream.of(indexes.computeIfAbsent(ordinal,
                                    ordKey -> IntStream.range(0, offsets.get(ordKey+1))
                                            .map(ind -> ind + offsets.get(ordKey))
                                            .toArray()))
                    ).toArray());
        }

        private void reset() {
            indexes.clear();
        }
    }

    private final List<TransferTask> tasksToMerge;
    private final TransferTask dependentTask;
    private final List<DimIndexBuffer> sourceIndexBuffer;
    private final List<DimIndexBuffer> targetIndexBuffer;

    /**
     * Create a builder for this class
     *
     * @return a new {@link Builder} instance.
     */
    public static Builder builder() {
        return new Builder();
    }


    public MergeTransferTask(
            List<TransferTask> tasksToMerge,
            List<DimIndexBuffer> sourceIndexBuffer,
            List<DimIndexBuffer> targetIndexBuffer,
            TransferTask dependentTask) {
        this.tasksToMerge = tasksToMerge;
        this.dependentTask = dependentTask;
        this.sourceIndexBuffer = sourceIndexBuffer;
        this.targetIndexBuffer = targetIndexBuffer;
    }

    @Override
    public void addWantedElementsFromSource(int dim, int[] indexes) {
        throw new UnsupportedOperationException("Dependent task mode not supported!");
    }

    @Override
    public void addWantedElementsFromTarget(int dim, int[] indexes) {
        throw new UnsupportedOperationException("Dependent task mode not supported!");
    }

    @Override
    public void addWantedNrofElementsFromTarget(int dim, int nrofElements) {
        throw new UnsupportedOperationException("Dependent task mode not supported!");
    }

    @Override
    public void execute() {
        tasksToMerge.forEach(TransferTask::execute);
        transferBufferedIndexes();
    }

    public void transferBufferedIndexes() {
        sourceIndexBuffer.forEach(
                dimIndexBuffer -> dimIndexBuffer.getAllIndexes()
                        .ifPresent(indexes -> dependentTask.addWantedElementsFromSource(dimIndexBuffer.dimension, indexes)));

        targetIndexBuffer.forEach(
                dimIndexBuffer -> dimIndexBuffer.getAllIndexes()
                        .ifPresent(indexes -> dependentTask.addWantedElementsFromTarget(dimIndexBuffer.dimension, indexes)));

        sourceIndexBuffer.forEach(DimIndexBuffer::reset);
        targetIndexBuffer.forEach(DimIndexBuffer::reset);
    }

    public static class Builder implements ListBuilder {

        private final List<TransferTask.ListBuilder> mergedInputs = new ArrayList<>();
        private List<DimIndexBuffer> sourceShapes;
        private List<DimIndexBuffer> targetShapes;
        private Optional<ListBuilder> dependentTaskBuilder = Optional.empty();


        /**
         * Adds a new input which is to be merged into the dependent task
         *
         * @param sourceShape  Shape of the source
         * @param targetShape  Shape of the target
         * @param inputBuilder Builder for the input
         * @return This Builder
         */
        Builder addInput(long[] sourceShape, long[] targetShape, TransferTask.ListBuilder inputBuilder) {
            final int ordinal = mergedInputs.size();
            if (ordinal == 0) {
                sourceShapes = IntStream.range(0, sourceShape.length)
                        .mapToObj(DimIndexBuffer::new)
                        .collect(Collectors.toList());
                targetShapes = IntStream.range(0, targetShape.length)
                        .mapToObj(DimIndexBuffer::new)
                        .collect(Collectors.toList());
            }

            IntStream.range(0, sourceShape.length).forEach(dim -> sourceShapes.get(dim).addOffset(sourceShape[dim]));
            IntStream.range(0, targetShape.length).forEach(dim -> targetShapes.get(dim).addOffset(targetShape[dim]));

            final CallbackTransferTask.Builder bufferingBuilder = CallbackTransferTask.builder()
                    .setSourceCallback((dim, indexes) -> sourceShapes.get(dim).addIndexes(ordinal, indexes))
                    .setTargetCallback((dim, indexes) -> targetShapes.get(dim).addIndexes(ordinal, indexes));
            inputBuilder.addDependentTask(bufferingBuilder);
            mergedInputs.add(inputBuilder);
            return this;
        }

        @Override
        public Builder addDependentTask(ListBuilder dependentTaskBuilder) {
            if (!this.dependentTaskBuilder.isPresent()) {
                this.dependentTaskBuilder = Optional.of(dependentTaskBuilder);
            } else {
                this.dependentTaskBuilder.get().addDependentTask(dependentTaskBuilder);
            }
            return this;
        }

        @Override
        public MergeTransferTask build() {

            return new MergeTransferTask(
                    mergedInputs.stream().map(ListBuilder::build).collect(Collectors.toList()),
                    sourceShapes,
                    targetShapes,
                    dependentTaskBuilder.map(ListBuilder::build).orElse(new NoTransferTask()));
        }
    }
}
