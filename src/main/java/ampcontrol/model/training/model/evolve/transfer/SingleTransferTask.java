package ampcontrol.model.training.model.evolve.transfer;

import lombok.Builder;
import lombok.Getter;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;

import java.util.Comparator;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

/**
 * Task to transfer weights from a source to a target {@link IndMapping}, typically when source and target
 * have different sizes.
 *
 * @author Christian Skärby
 */
@Builder(builderClassName = "Builder", buildMethodName = "buildInternal")
public class SingleTransferTask implements TransferTask {

    @lombok.Builder.Default
    private final IndMapping source = IndMapping.builder().build();
    @lombok.Builder.Default
    private final IndMapping target = IndMapping.builder().build();
    private final Function<Integer, Comparator<Integer>> compFactory;
    @lombok.Singular("maskDim") private final Set<Integer> dimensionMask;
    @lombok.Builder.Default private final TransferTask dependentTask = new NoTransferTask();


    @lombok.Builder(builderClassName = "Builder")
    public static class IndMapping {

        @Getter
        private final TransferRegistry.ArrayEntry entry;
        @lombok.Builder.Default
        private final IntUnaryOperator dimensionMapper = IntUnaryOperator.identity();
        @lombok.Builder.Default
        private final IntFunction<IntUnaryOperator> remapper = dim -> IntUnaryOperator.identity();

        void addWantedElements(int dim, int[] wantedElementInds) {
            final int mappedDim = dimensionMapper.applyAsInt(dim);
            final INDArrayIndex newIndex = new SpecifiedIndex(IntStream.of(wantedElementInds).map(remapper.apply(mappedDim)).toArray());
            entry.addIndArrayIndex(mappedDim, newIndex);
        }

        void addWantedNrofElements(int dim, int nrofElements) {
            final int mappedDim = dimensionMapper.applyAsInt(dim);
            final INDArrayIndex newIndex = NDArrayIndex.interval(remapper.apply(mappedDim).applyAsInt(0), remapper.apply(mappedDim).applyAsInt(nrofElements));
            entry.addIndArrayIndex(mappedDim, newIndex);
        }
    }

    @Override
    public void addWantedElementsFromSource(int dim, int[] indexes) {
        if(!dimensionMask.contains(dim)) {
            source.addWantedElements(dim, indexes);
        }
        dependentTask.addWantedElementsFromSource(dim, indexes);
    }

    @Override
    public void addWantedNrofElementsFromTarget(int dim, int nrofElements) {
        if(!dimensionMask.contains(dim)) {
            target.addWantedNrofElements(dim, nrofElements);
        }
        dependentTask.addWantedNrofElementsFromTarget(dim, nrofElements);
    }

    @Override
    public void execute() {
        final long[] sourceShape = source.getEntry().shape();
        final long[] targetShape = target.getEntry().shape();

        for (int i = 0; i < targetShape.length; i++) {
            if(dimensionMask.contains(i)) {
                continue;
            }

            if (targetShape[i] < sourceShape[i]) {
                final int tensorDim = i;

                final Comparator<Integer> comparator = compFactory.apply(tensorDim);
                final int[] wantedElements = IntStream.range(0, (int) sourceShape[tensorDim])
                        .boxed()
                        .sorted(comparator)
                        .limit(targetShape[tensorDim])
                        .mapToInt(e -> e)
                        .sorted()
                        .toArray();

                addWantedElementsFromSource(tensorDim, wantedElements);
            }

            if (targetShape[i] > sourceShape[i]) {
                addWantedNrofElementsFromTarget(i, (int) sourceShape[i]);
            }
        }
    }



    public static class Builder implements ListBuilder {
        // Used through lombok
        private Function<Integer, Comparator<Integer>> compFactory = dim -> source.getEntry().defaultComparatorFactory(dim);

        private Optional<ListBuilder> dependentTaskBuilder = Optional.empty();

        // To trick lombok so all boilerplate is done in autogenerated buildInternal
        public SingleTransferTask build() {
            target.getEntry().put(source.getEntry());
            dependentTaskBuilder.map(ListBuilder::build).ifPresent(this::dependentTask);
            return this.buildInternal();
        }

        /**
         * Adds a {@link Builder} for a dependent {@link TransferTask}. If a builder is already set, the added builder
         * will be added as a dependent builder for the existing builder, thus creating a linked list of dependent tasks
         * @param dependentTaskBuilder the dependent task
         * @return This builder
         */
        public Builder addDependentTask(ListBuilder dependentTaskBuilder) {
            if(!this.dependentTaskBuilder.isPresent()) {
                this.dependentTaskBuilder = Optional.of(dependentTaskBuilder);
            } else {
                this.dependentTaskBuilder.get().addDependentTask(dependentTaskBuilder);
            }
            return this;
        }

        private Builder dependendTask(TransferTask dependentTask) {
            this.dependentTask = dependentTask;
            return this;
        }
    }
}
