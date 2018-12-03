package ampcontrol.model.training.model.evolve.transfer;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Buffers transferred indexes from several inputs and merges the result into a dependent {@link TransferTask}. Inputs
 * must be added in the same order as which they are concatenated.
 * <br><br>
 * For example, if A and B are both inputs to C so that input to C is [A, B], then {@link TransferTask.ListBuilder} for
 * A must be added before {@link TransferTask.ListBuilder} for B. The {@link TransferTask} for C is the dependent
 * {@link TransferTask}.
 *
 * @author Christian Skärby
 */
public final class MergeTransferBuffer {

    private final TransferTask dependentTask;
    private final List<DimIndexBuffer> sourceIndexBuffer;
    private final List<DimIndexBuffer> targetIndexBuffer;

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

    /**
     * Create a builder for this class
     *
     * @return a new {@link Builder} instance.
     */
    public static Builder builder() {
        return new Builder();
    }


    public MergeTransferBuffer(
            List<DimIndexBuffer> sourceIndexBuffer,
            List<DimIndexBuffer> targetIndexBuffer,
            TransferTask dependentTask) {
        this.dependentTask = dependentTask;
        this.sourceIndexBuffer = sourceIndexBuffer;
        this.targetIndexBuffer = targetIndexBuffer;
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

    public static class Builder {

        private int ordinal = 0;
        private List<DimIndexBuffer> sourceShapes;
        private List<DimIndexBuffer> targetShapes;
        private TransferTask.ListBuilder dependentTaskBuilder = NoTransferTask.builder();


        /**
         * Adds a new input which is to be merged into the dependent task
         *
         * @param sourceShape  Shape of the source
         * @param targetShape  Shape of the target
         * @param inputBuilder Builder for the input
         * @return This Builder
         */
        Builder addInput(long[] sourceShape, long[] targetShape, TransferTask.ListBuilder inputBuilder) {
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

            final int thisOrdinal = ordinal;
            final CallbackTransferTask.Builder bufferingBuilder = CallbackTransferTask.builder()
                    .setSourceCallback((dim, indexes) -> sourceShapes.get(dim).addIndexes(thisOrdinal, indexes))
                    .setTargetCallback((dim, indexes) -> targetShapes.get(dim).addIndexes(thisOrdinal, indexes));
            inputBuilder.addDependentTask(bufferingBuilder);
            ordinal++;
            return this;
        }

        public Builder addDependentTask(TransferTask.ListBuilder dependentTaskBuilder) {
            this.dependentTaskBuilder = this.dependentTaskBuilder.addDependentTask(dependentTaskBuilder);
            return this;
        }

        public MergeTransferBuffer build() {
            return new MergeTransferBuffer(
                    sourceShapes,
                    targetShapes,
                    dependentTaskBuilder.build());
        }
    }
}
